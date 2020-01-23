#!/usr/bin/env python3
"""
Sample application: skeleton for new applications

"""
import argparse
import csv
import logging
import pkg_resources
import shutil
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import hunspell
import pandas as pd
import textacy
from sklearn.feature_extraction.text import TfidfVectorizer
from strictyaml import YAMLValidationError
from textacy import preprocessing

from ana_consult.ac_conf import AnaConsultConf

from . import _, __version__

APP_NAME = "mtes_analyze"

logger = logging.getLogger(APP_NAME)

# Spell chacking word counter (global)
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


def preprocess(config: str):
    """Load raw csv file from scraper and do first processing."""
    logger = logging.getLogger(APP_NAME + ".preprocess")
    pd.set_option("display.max_colwidth", 40)
    csv_file = Path.home() / (
        "ana_consult/data/raw/" + config.consultation_name + ".csv"
    )
    logger.info(_("Loading %s"), csv_file)
    data = pd.read_csv(csv_file, header=0, quoting=csv.QUOTE_ALL)
    logger.info(_("Loaded %s rows of raw data"), len(data))

    # Split subject in specific fields
    data[["titre", "nom", "date", "heure"]] = data.sujet.str.extract(
        "(.*), par  (.*) ,, le (.*) à (.*)", expand=True
    )
    data = data.drop(columns=["sujet"])
    data = data[["titre", "nom", "date", "heure", "texte"]]
    csv_file = Path.home() / (
        "ana_consult/data/interim/" + config.consultation_name + "_split.csv"
    )
    data.to_csv(csv_file, index=False, quoting=csv.QUOTE_ALL)

    # Drop duplicated lines
    # print(repr(data[["titre", "nom", "texte"]].head(10)))
    data.drop_duplicates(subset=["nom", "texte"], inplace=True)
    # Drop lines in English (containing wolf)
    data.drop(data[data.texte.str.contains("wolf", case=False)].index, inplace=True)
    logger.info(_("Storing %s rows of pre-processed data"), len(data))
    # print(data[["titre", "nom", "texte"]].head(12))
    csv_file = Path.home() / (
        "ana_consult/data/interim/" + config.consultation_name + "_prep.csv"
    )
    data.to_csv(csv_file, index=False, quoting=csv.QUOTE_ALL)


# Spell correction of misspelled words
def spell_correction(spell, doc, logger):
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


def process(config: str):
    """Load pre-processed csv file do first base NLP processing."""
    logger = logging.getLogger(APP_NAME + ".process")
    pd.set_option("display.max_colwidth", 120)
    csv_file = Path.home() / (
        "ana_consult/data/interim/" + config.consultation_name + "_prep.csv"
    )
    logger.info(_("Loading %s"), csv_file)
    responses = pd.read_csv(csv_file, header=0, quoting=csv.QUOTE_ALL, nrows=100000)
    # Merge in one text column
    responses["raw_text"] = responses["titre"] + ". " + responses["texte"]
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
        preprocessing.replace.replace_currency_symbols, replace_with=""
    )

    # Prepare NLP processing
    logger.info(_("NLP text processing"))
    fr_nlp = textacy.load_spacy_lang(
        "fr_core_news_sm", disable=("tagger", "parser", "ner")
    )
    logger.info(_("NLP pipeline: %s"), fr_nlp.pipe_names)
    # Adjust stopwords for this specific topic
    fr_nlp.Defaults.stop_words |= {
        "y",
        "france",
        "italie",
    }
    fr_nlp.Defaults.stop_words -= {"contre", "pour"}
    # Prepare first corpus from raw text, for spell checking
    corpus = textacy.Corpus(fr_nlp)
    [corpus.add_text(t) for t in responses["raw_text"]]
    logger.info(_("Response raw corpus %s"), corpus)

    # Correct spelling and remove stopwords, ponctuation and spaces
    logger.info(_("Spell checking NLP document"))
    spell = hunspell.HunSpell(
        "/usr/share/hunspell/fr_FR.dic", "/usr/share/hunspell/fr_FR.aff"
    )
    added_words = pkg_resources.resource_filename(__name__, "data/mtes.txt")
    spell.add_dic(added_words)
    responses["checked_text"] = ""
    for d in range(corpus.n_docs):
        responses["checked_text"][d] = spell_correction(spell, corpus[d], logger)

    # for row in responses.head(n=2).itertuples():
    #     print("--------------------------")
    #     print(row.titre, row.nom, row.date, row.heure)
    #     print(row.raw_text)
    #     print(row.checked_text)

    # Prepare final corpus from spell-checked text, for analysis
    corpus = textacy.Corpus(fr_nlp)
    for row in responses.itertuples():
        corpus.add_record(
            (row.checked_text, {"name": row.nom, "date": row.date, "time": row.heure})
        )
    logger.info(_("Response spell checked corpus %s"), corpus)

    # Save data
    pkl_file = Path.home() / (
        "ana_consult/data/interim/" + config.consultation_name + "_doc.pkl"
    )
    logger.info(_("Storing NLP document to %s"), pkl_file)
    corpus.save(pkl_file)


def cluster(config: str):
    """Perform clustering on NLP processed data."""
    logger = logging.getLogger(APP_NAME + ".cluster")
    # Load data
    pkl_file = Path.home() / (
        "ana_consult/data/interim/" + config.consultation_name + "_doc.pkl"
    )
    logger.info(_("Loading NLP document from %s"), pkl_file)
    doc = pd.read_pickle(pkl_file)
    logger.info(_("Document size: %s"), doc.shape)
    # define vectorizer parameters
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.8, min_df=0.2, stop_words=None, use_idf=True, ngram_range=(1, 3),
    )
    # Fit vectoriser to NLP processed column
    logger.info(_("Fitting TF-IDF vectorizer to NLP data"))
    tfidf_matrix = tfidf_vectorizer.fit_transform(doc)
    terms = tfidf_vectorizer.get_feature_names()
    logger.info(_("TF-IDF (n_samples, n_features): %s"), tfidf_matrix.shape)
    corpus_index = [n for n in doc]
    df = pd.DataFrame(tfidf_matrix.todense(), index=corpus_index, columns=terms)
    print(df[0:10])


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

    # Preprocess csv file
    if args.preprocess:
        logger.info(_("Preprocessing raw csv file"))
        preprocess(ac_ctrl)
        return None

    # NLP process csv file
    if args.process:
        logger.info(_("NLP processing pre-processed csv file"))
        process(ac_ctrl)
        return None

    # Clustering
    if args.cluster:
        logger.info(_("Clustering processed file"))
        cluster(ac_ctrl)
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
