import argparse


def get_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="RealSense Camera Checkerboard Detection"
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=10,
        help="Number of columns in the checkerboard pattern",
    )
    parser.add_argument(
        "--rows", type=int, default=7, help="Number of rows in the checkerboard pattern"
    )
    parser.add_argument(
        "--square-size", type=float, default=19.09, help="Size of each square in mm"
    )
    return parser.parse_args()
