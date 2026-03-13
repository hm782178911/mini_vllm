import logging

def get_logger(name):

    logging.basicConfig(
        # level=logging.INFO,
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    return logging.getLogger(name)