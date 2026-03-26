
"""
logger.py

Module de journalisation centralisé pour le pipeline RAG.

Fonctionnalités :
- Logs console + fichier
- 1 fichier par jour
- Rotation automatique si taille dépassée
- Gestion des erreurs (fail-safe)
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "rag_pipeline") -> logging.Logger:
    """
    Initialise et configure un logger robuste.

    Features :
    - Console + fichier
    - Fichier journalier (YYYY-MM-DD.log)
    - Rotation automatique (max 5 MB, 3 backups)
    - Protection contre duplication handlers
    - Gestion des erreurs interne (fail-safe)

    Args:
        name (str): Nom du logger

    Returns:
        logging.Logger: Logger configuré
    """

    logger = logging.getLogger(name)

    # Éviter duplication handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    try:
        # -----------------------------
        # FORMAT
        # -----------------------------
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        # -----------------------------
        # CONSOLE HANDLER
        # -----------------------------
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # -----------------------------
        # DOSSIER LOGS
        # -----------------------------
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # -----------------------------
        # NOM FICHIER (DATE)
        # -----------------------------
        today_str = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{today_str}.log"

        # -----------------------------
        # FILE HANDLER AVEC ROTATION
        # -----------------------------
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,             # 3 fichiers max (ex: .1, .2, .3)
            encoding="utf-8"
        )

        file_handler.setFormatter(formatter)

        # -----------------------------
        # AJOUT HANDLERS
        # -----------------------------
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        logger.info("Logger initialisé avec succès")

    except Exception as e:
        # Fail-safe : ne jamais casser l'application
        print(f"❌ Erreur initialisation logger : {e}")

    return logger
