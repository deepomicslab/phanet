"""Product annotation mapping to the nine Table S3 categories."""

from phanet.datasets.phagescope_s3 import CATEGORIES, PhageScopeS3Mapper, VERSION

MAPPING_VERSION = VERSION


class AnnotationMapper(PhageScopeS3Mapper):
    """Map protein product descriptions to functional categories."""

    pass
