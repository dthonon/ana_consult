#!/usr/bin/env python3
"""
Sample application: skeleton for new applications

"""
import argparse
import csv
import hashlib
import logging
import pkg_resources
import re
import shutil
import sys
import unicodedata
from collections import Counter
from functools import lru_cache
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import hunspell
# import numpy as np
import pandas as pd
import textacy
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from strictyaml import YAMLValidationError
from textacy import preprocessing

from ana_consult.ac_conf import AnaConsultConf

from . import _, __version__

APP_NAME = "mtes_analyze"

logger = logging.getLogger(APP_NAME)

# Spell checking word counter (global)
nb_words = 0


def arguments(args):
    """Define and parse command arguments.

    Args:
        args ([str]): command line parameters as list of strings

    Returns:
        :obj:`argparse.Namespace`: command line parameters namespace
    """
    # Get options
    parser = argparse.ArgumentParser(
        description="Sample Biolovision API client application."
    )
    parser.add_argument(
        "--version",
        help=_("Print version number"),
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument(
        "--verbose", help=_("Increase output verbosity"), action="store_true"
    )
    out_group.add_argument(
        "--quiet", help=_("Reduce output verbosity"), action="store_true"
    )
    parser.add_argument(
        "--init", help=_("Initialize the YAML configuration file"), action="store_true"
    )
    parser.add_argument(
        "--preprocess",
        help=_("Load raw csv file from scraper and do first processing"),
        action="store_true",
    )
    parser.add_argument(
        "--process",
        help=_("Load pre-processed csv file and do basinc NLP processing"),
        action="store_true",
    )
    parser.add_argument(
        "--cluster",
        help=_("Load raw csv file from scraper and do first processing"),
        action="store_true",
    )
    parser.add_argument("config", help=_("Configuration file name"))

    return parser.parse_args(args)


def init(config: str):
    """Copy template YAML file to home directory."""
    logger = logging.getLogger(APP_NAME + ".init")
    yaml_src = pkg_resources.resource_filename(__name__, "data/cfg_template.yaml")
    yaml_dst = str(Path.home() / config)
    logger.info(_("Creating YAML configuration file %s, from %s"), yaml_dst, yaml_src)
    shutil.copyfile(yaml_src, yaml_dst)
    logger.info(_("Please edit %s before running the script"), yaml_dst)


class Consultation(object):
    """Provides methods to process a consultation."""

    def __init__(self):
        super().__init__()
        pd.set_option("display.max_colwidth", 40)

    @property
    @lru_cache(maxsize=1)
    def _fr_nlp(self):
        logger = logging.getLogger(APP_NAME + ".__init__")
        # Prepare NLP processing
        logger.info(_("Preparing NLP text processing"))
        _nlp = textacy.load_spacy_lang(
            "fr_core_news_sm", disable=("tagger", "parser", "ner")
        )
        logger.info(_("NLP pipeline: %s"), _nlp.pipe_names)
        # Adjust stopwords for this specific topic
        _nlp.Defaults.stop_words |= {"y", "france", "italie"}
        _nlp.Defaults.stop_words -= {"contre"}
        return _nlp

    def _clean_unicode(self, ch):
        """Remove unprintable character
        You can find a full list of categories here:
        http://www.fileformat.info/info/unicode/category/index.htm"""
        letters = ("LC", "Ll", "Lm", "Lo", "Lt", "Lu")
        numbers = ("Nd", "Nl", "No")
        marks = ("Mc", "Me", "Mn")
        punctuation = ("Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps")
        symbol = ("Sc", "Sk", "Sm", "So")
        space = ("Zl", "Zp", "Zs")

        allowed_categories = letters + numbers + marks + punctuation + symbol
        if unicodedata.category(ch) in space:
            return " "
        elif unicodedata.category(ch) in allowed_categories:
            return ch
        else:
            return " "

    def _remove_tags(self, text):
        """Cleans a string :
        - removing HTML tags
        - removing shortcuts, such as =>...
        - removing non printable text, based on a whitelist of printable unicode
        """
        TAG_RE = re.compile(r"<[^>]+>|[=\-,]?>|\+|\/\/")
        text = TAG_RE.sub("", text)

        text = u"".join([self._clean_unicode(c) for c in text])
        return text

    def preprocess(self, config: str):
        """Load raw csv file from scraper and do first processing:
        - Split subject in title, name, datetime.
        - Drop duplicates."""
        logger = logging.getLogger(APP_NAME + ".preprocess")
        csv_file = Path.home() / (
            "ana_consult/data/raw/" + config.consultation_name + ".csv"
        )
        logger.info(_("Loading %s"), csv_file)
        responses = pd.read_csv(
            csv_file, header=0, quoting=csv.QUOTE_ALL, nrows=1000000
        )
        logger.info(_("Loaded %s rows of raw data"), len(responses))

        # Add hash-key to ensure cross-reference with manually modified files
        responses["uid"] = responses.sujet.apply(
            lambda t: hashlib.sha224(t.encode("utf-8")).hexdigest()
        )

        # Split subject in specific fields
        responses[["titre", "nom", "date", "heure"]] = responses.sujet.str.extract(
            "(.*), par  (.*) ,, le (.*) à (.*)", expand=True
        )
        responses = responses.drop(columns=["sujet"])
        responses = responses[["titre", "nom", "date", "heure", "texte", "uid"]]

        # Drop duplicated lines
        responses.drop_duplicates(subset=["nom", "titre"], inplace=True)
        responses.drop_duplicates(subset=["nom", "texte"], inplace=True)
        logger.info(_("%s rows remaining after deduplication"), len(responses))

        # Merge in one text column
        responses["raw_text"] = responses["titre"] + ". " + responses["texte"]
        responses["raw_text"].fillna(value="?", inplace=True)
        responses = responses.drop(columns=["texte"])

        # Cleanup
        responses["raw_text"] = responses["raw_text"].apply(
            preprocessing.normalize.normalize_whitespace
        )
        responses["raw_text"] = responses["raw_text"].apply(
            preprocessing.replace.replace_urls, replace_with=""
        )
        responses["raw_text"] = responses["raw_text"].apply(
            preprocessing.replace.replace_numbers, replace_with=""
        )
        responses["raw_text"] = responses["raw_text"].apply(
            preprocessing.replace.replace_emojis, replace_with=""
        )
        responses["raw_text"] = responses["raw_text"].apply(
            preprocessing.replace.replace_currency_symbols, replace_with="Euros"
        )
        responses["raw_text"] = responses["raw_text"].apply(self._remove_tags)

        # Save preprocessed data to csv file
        logger.info(_("Storing %s rows of pre-processed data"), len(responses))
        csv_file = Path.home() / (
            "ana_consult/data/interim/" + config.consultation_name + "_prep.csv"
        )
        responses.to_csv(csv_file, index=False, quoting=csv.QUOTE_ALL)

        # Merge with category file, if exists
        csv_file = Path.home() / (
            "ana_consult/data/external/" + config.consultation_name + "_cat.csv"
        )
        logger.info(_("Loading category file %s"), csv_file)
        categories = pd.read_csv(
            csv_file, header=0, quoting=csv.QUOTE_ALL, nrows=1000000
        )
        logger.info(_("Loaded %s rows of category data"), len(categories))

    @staticmethod
    def _spell_correction(spell, doc, logger):
        """Spell correction of misspelled words."""
        global nb_words
        nb_words += 1
        if (nb_words % 100) == 0:
            logger.info(_("Spell checking word number %d"), nb_words)
        text = ""
        for d in doc:
            word = d.text
            # Spell check meaningfull words only
            if d.is_space:
                pass  # Nothing to check
            elif d.is_stop or d.is_punct or spell.spell(word):
                text += d.text_with_ws
            else:
                sp = spell.suggest(word)
                if len(sp) > 0:
                    print(word + " => " + sp[0])
                    text += sp[0] + d.whitespace_
                else:
                    logger.warning(_("Unable to correct %s"), word)
                    text += d.text_with_ws
        return text

    def process(self, config: str):
        """Load pre-processed csv file do first base NLP processing:
        - Clean text from unusable element (url, non-printable...)
        - Spelling correction"""
        logger = logging.getLogger(APP_NAME + ".process")
        # pd.set_option("display.max_colwidth", 120)
        # Read preprocessed data
        csv_file = Path.home() / (
            "ana_consult/data/interim/" + config.consultation_name + "_prep.csv"
        )
        logger.info(_("Loading data %s"), csv_file)
        responses = pd.read_csv(csv_file, header=0, quoting=csv.QUOTE_ALL, nrows=500)
        logger.info(_("Loaded %s rows of pre-processed data"), len(responses))
        # Read classification data, if it exists
        csv_file = Path.home() / (
            "ana_consult/data/external/" + config.consultation_name + "_cat.csv"
        )
        if csv_file.exists():
            logger.info(_("Loading classification %s"), csv_file)
            classif = pd.read_csv(
                csv_file, header=0, quoting=csv.QUOTE_ALL, nrows=1000000
            )
            logger.info(_("Loaded %s rows of classification data"), len(classif))
            classif = classif.drop(columns=["raw_text"])
            responses = responses.merge(classif, on="uid")
        else:
            logger.info(_("No classification found %s"), csv_file)
            classif = None

        # Prepare first corpus from raw text, for spell checking
        corpus = textacy.Corpus(
            self._fr_nlp,
            data=(
                t
                for t in responses["raw_text"]
                if textacy.lang_utils.identify_lang(t) == "fr"
            ),
        )
        logger.info(_("Response pre-processed corpus %s"), corpus)

        # Correct spelling and remove stopwords, ponctuation and spaces
        logger.info(_("Spell checking NLP document"))
        spell = hunspell.HunSpell(
            "/usr/share/hunspell/fr_FR.dic", "/usr/share/hunspell/fr_FR.aff"
        )
        added_words = pkg_resources.resource_filename(__name__, "data/mtes.txt")
        spell.add_dic(added_words)
        spell.remove("abatage")
        responses["checked_text"] = ""
        for d in range(corpus.n_docs):
            responses["checked_text"][d] = self._spell_correction(
                spell, corpus[d], logger
            )

        # for row in responses.head(n=2).itertuples():
        #     print("--------------------------")
        #     print(row.titre, row.nom, row.date, row.heure)
        #     print(row.raw_text)
        #     print(row.checked_text)

        # Prepare final corpus from spell-checked text, for analysis
        corpus = textacy.Corpus(self._fr_nlp)
        for row in responses.itertuples():
            corpus.add_record(
                (
                    row.checked_text,
                    {
                        "name": row.nom,
                        "date": row.date,
                        "time": row.heure,
                        "opinion": row.opinion,
                        "uid": row.uid,
                    },
                )
            )
        logger.info(_("Response spell checked corpus %s"), corpus)
        # print(corpus[0]._.preview)
        # print("meta:", corpus[0]._.meta)
        # Save data
        corpus_file = Path.home() / (
            "ana_consult/data/interim/" + config.consultation_name + "_doc.pkl"
        )
        logger.info(_("Storing NLP document to %s"), corpus_file)
        corpus.save(corpus_file)

    def cluster(self, config: str):
        """Perform clustering on NLP processed data."""
        logger = logging.getLogger(APP_NAME + ".cluster")
        # Load data
        corpus_file = Path.home() / (
            "ana_consult/data/interim/" + config.consultation_name + "_doc.pkl"
        )
        logger.info(_("Loading corpus from %s"), corpus_file)
        corpus = textacy.Corpus.load(self._fr_nlp, corpus_file)
        logger.info(_("Document size: %s"), corpus)

        # Define vectorizer parameters
        logger.info(_("Simplifying corpus"))
        doc_lemma = pd.DataFrame(
            [
                [
                    " ".join(
                        list(
                            doc._.to_terms_list(
                                ngrams=1,
                                entities=False,
                                normalize="lemma",
                                as_strings=True,
                                filter_stops=True,
                                filter_punct=True,
                                filter_nums=True,
                            )
                        )
                    ),
                    doc._.meta["opinion"],
                ]
                for doc in corpus[:1000000]
            ],
            columns=["text", "opinion"],
        ).dropna()
        print(doc_lemma.head(20))

        tfidf_vectorizer = TfidfVectorizer(
            max_df=0.9, min_df=0.1, stop_words=None, use_idf=True, ngram_range=(1, 3)
        )
        # Fit vectoriser to NLP processed column
        logger.info(_("Fitting TF-IDF vectorizer to NLP data"))
        tfidf_matrix = tfidf_vectorizer.fit_transform(doc_lemma.text)
        terms = tfidf_vectorizer.get_feature_names()
        logger.info(_("TF-IDF (n_samples, n_features): %s"), tfidf_matrix.shape)
        # corpus_index = [n for n in doc_lemma]
        # df = pd.DataFrame(tfidf_matrix.todense(), index=corpus_index, columns=terms)
        # print(df[0:10])

        # K-means clustering
        logger.info(_("K-means clustering"))
        true_k = 2
        model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=1)
        pred_labels = list(model.fit_predict(tfidf_matrix))
        logger.info(_("Cluster summary:"))
        true_labels = [0 if d == "Favorable" else 1 for d in doc_lemma.opinion]
        print(true_labels[:50])
        print(pred_labels[:50])
        print(
            "Homogeneity: %0.3f" % metrics.homogeneity_score(true_labels, pred_labels)
        )
        print(metrics.confusion_matrix(true_labels, pred_labels))
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = tfidf_vectorizer.get_feature_names()
        cl_size = Counter(model.labels_)
        for i in range(true_k):
            logger.info(
                _("Cluster %d, proportion: %d%%, top terms:"),
                i,
                cl_size[i] / len(corpus) * 100,
            )
            top_t = ", ".join([terms[t] for t in order_centroids[i, :10]])
            logger.info(top_t)

        # # DBSCAN clustering
        # logger.info(_("DBSCAN clustering"))
        # model = DBSCAN(eps=2.1, min_samples=10, metric="l1")
        # model.fit(tfidf_matrix)
        # logger.info(_("Cluster summary:"))
        # core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
        # core_samples_mask[model.core_sample_indices_] = True
        # labels = model.labels_
        # # Number of clusters in labels, ignoring noise if present.
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # n_noise_ = list(labels).count(-1)
        # logger.info(_("Estimated number of clusters: %d"), n_clusters_)
        # logger.info(_("Estimated number of noise points: %d"), n_noise_)
        # for i in range(n_clusters_):
        #     logger.info(
        #         _("Cluster %d, proportion: %d%%"), i, cl_size[i] / len(corpus) * 100
        #     )


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    # Create $HOME/tmp directory if it does not exist
    (Path.home() / "tmp").mkdir(exist_ok=True)

    # Define logger format and handlers
    logger = logging.getLogger(APP_NAME)
    # Create file handler which logs even debug messages
    fh = TimedRotatingFileHandler(
        str(Path.home()) + "/tmp/" + APP_NAME + ".log",
        when="midnight",
        interval=1,
        backupCount=100,
    )
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Get command line arguments
    args = arguments(args)

    # Define verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.info(_("%s, version %s"), sys.argv[0], __version__)
    logger.info(_("Arguments: %s"), sys.argv[1:])

    # If required, first create YAML file
    if args.init:
        logger.info(_("Creating YAML configuration file"))
        init(args.config)
        return None

    # Get configuration from file
    if not (Path.home() / args.config).is_file():
        logger.critical(
            _("Configuration file %s does not exist"), str(Path.home() / args.config)
        )
        return None
    logger.info(_("Getting configuration data from %s"), args.config)
    try:
        ac_ctrl = AnaConsultConf(args.config)
    except YAMLValidationError:
        logger.critical(_("Incorrect content in YAML configuration %s"), args.config)
        sys.exit(0)

    # Create consultation object
    consult = Consultation()

    # Preprocess csv file
    if args.preprocess:
        logger.info(_("Preprocessing raw csv file"))
        consult.preprocess(ac_ctrl)
        return None

    # NLP process csv file
    if args.process:
        logger.info(_("NLP processing pre-processed csv file"))
        consult.process(ac_ctrl)
        return None

    # Clustering
    if args.cluster:
        logger.info(_("Clustering processed file"))
        consult.cluster(ac_ctrl)
        return None

    logger.info(_("End of processing"))
    return None


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


# Main wrapper
if __name__ == "__main__":
    run()
