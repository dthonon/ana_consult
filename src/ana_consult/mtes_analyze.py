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

import pandas as pd
from strictyaml import YAMLValidationError

from ana_consult.ac_conf import AnaConsultConf

from . import _, __version__

APP_NAME = "mtes_analyze"

logger = logging.getLogger(APP_NAME)


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
    pd.set_option("display.max_colwidth", 120)
    csv_file = Path.home() / ("tmp/" + config.consultation_name + ".csv")
    logger.info(_("Loading %s"), csv_file)
    data = pd.read_csv(csv_file, header=0, quoting=csv.QUOTE_ALL)
    print(data[["sujet"]].head(60))
    data[["titre", "nom", "date", "heure"]] = data.sujet.str.extract(
        "(.*), par  (.*) ,, le (.*) à (.*)", expand=True
    )
    data = data.drop(columns=["sujet"])
    data = data[["titre", "nom", "date", "heure", "texte"]]
    print(data[["titre"]].head(60))
    csv_file = Path.home() / ("tmp/" + config.consultation_name + "_prep.csv")
    data.to_csv(csv_file, index=False, quoting=csv.QUOTE_ALL)


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    # Create $HOME/tmp directory if it does not exist
    (Path.home() / "tmp").mkdir(exist_ok=True)

    # Define logger format and handlers
    logger = logging.getLogger(APP_NAME)
    # create file handler which logs even debug messages
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

    return None


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


# Main wrapper
if __name__ == "__main__":
    run()
