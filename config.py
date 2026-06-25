from test import DatasetPaths

image_size = 224
num_classes = 2


dataset_paths = DatasetPaths(
    faceforensics_root="/path/to/FaceForensics++",
    dfd_real_dir="/path/to/DFD/original_sequences/actors/raw/videos",
    dfd_fake_dir="/path/to/DFD/fake/videos",
    dfdc_root="/path/to/DFDC",
    cdf_root="/path/to/CDF",
    dfdcp_root="/path/to/DFDCP/dfdc_preview_set",
    ffiw_root="/path/to/FFIW/FFIW-test/FFIW10K-v1-release-test",
)