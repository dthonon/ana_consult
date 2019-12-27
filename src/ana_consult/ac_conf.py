# pylint: disable=too-many-instance-attributes
"""Expose local configuration parameters as properties.

Parameters are defined in a YAML file located in $HOME directory.
This file is created using --init option, and then customized by the user.
Each time the application is run, this parameter file is read and the
parameters are then available as properties AnaConsultConf.

"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from strictyaml import (
    Email,
    Map,
    Optional,
    Str,
    Url,
    YAMLError,
    YAMLValidationError,
    load
)

from . import _, __version__

logger = logging.getLogger("ana_consult.ac_conf")


class AnaConsultConfException(Exception):
    """An exception occurred while loading parameters."""


class IncorrectParameter(AnaConsultConfException):
    """Incorrect or missing parameter."""


# Define PEP 484 types, TODO: refine type
_ConfType = Dict[str, Any]

# Define strictyaml schema
_ConfSchema = Map(
    {
        Optional("main"): Map({"admin_mail": Email()}),
        "consultation": Map({"site_url": Url(), "name": Str()}),
    }
)


class AnaConsultConf:
    """
    Read config file and expose properties
    """

    def __init__(self, file: str) -> None:
        # Define configuration schema
        # Read configuration parameters
        p = Path.home() / file
        yaml_text = p.read_text()
        try:
            logger.info(_("Loading YAML configuration %s"), file)
            self._config = load(yaml_text, _ConfSchema).data
        except YAMLValidationError:
            logger.critical(_("Incorrect content in YAML configuration %s"), file)
            logger.critical(_("%s"), sys.exc_info()[1])
            raise
        except YAMLError:  # pragma: no cover
            logger.critical(_("Error while reading YAML configuration %s"), file)
            raise

        # Import parameters in properties
        self._main_admin_mail = (
            ""
            if "admin_mail" not in self._config["main"]
            else self._config["main"]["admin_mail"]
        )  # type: str

        self._consultation_site_url = (
            ""
            if "site_url" not in self._config["consultation"]
            else self._config["consultation"]["site_url"]
        )  # type: str

        self._consultation_name = (
            ""
            if "name" not in self._config["consultation"]
            else self._config["consultation"]["name"]
        )  # type: str

    @property
    def version(self) -> str:
        """Return version."""
        return __version__

    @property
    def main_admin_mail(self) -> str:
        """Return property."""
        return self._main_admin_mail

    @property
    def consultation_site_url(self) -> str:
        """Return property."""
        return self._consultation_site_url

    @property
    def consultation_name(self) -> str:
        """Return property."""
        return self._consultation_name
