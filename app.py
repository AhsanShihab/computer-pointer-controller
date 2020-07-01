import cv2 
from argparse import ArgumentParser
from src.input_feeder import InputFeeder
from src.mouse_controller import MouseController

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m1", "--face", required=False, type=str, default = "model/intel/face-detection-adas-0001/FP16/face-detection-adas-0001",
                        help="Path to the face detection model IR files (without the extensions)")
    parser.add_argument("-m2", "--head", required=False, type=str, default = "model/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001",
                        help="Path to the head pose estimation model IR files (without the extensions)")
    parser.add_argument("-m3", "--landmark", required=False, type=str, default = "model/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009",
                        help="Path to the face landmark detection model IR files (without the extensions)")
    parser.add_argument("-m4", "--gaze", required=False, type=str, default = "model/intel/gaze-estimation-adas-0002/INT8/gaze-estimation-adas-0002",
                        help="Path to the gaze estimation model IR files (without the extensions)")
    parser.add_argument("-s", "--source", required=True, type=str, default = "video",
                        help="Define the inpute source, options: 'cam' for webcam input, 'video' for video file, 'image' for image file")
    parser.add_argument("-f", "--file", required=False, type=str, default = "bin/demo.mp4",
                        help="Path to the source file if source is not webcam")
    parser.add_argument("-d", "--device", required=False, type=str, default = "CPU",
                        help="The target device to run the inference on, options: CPU, GPU, VPU, FPGA")
    parser.add_argument("-e", "--extension", required=False, type=str,
                        default = "C:/Program Files (x86)/IntelSWTools/openvino_2019.3.379/deployment_tools/inference_engine/bin/intel64/Release/cpu_extension_avx2.dll",
                        help="If the used model(s) requires CPU extension, provide the location to the extension file")
    parser.add_argument("-r", "--results", required=False, type=int, default=-1,
                        help="Show the result from model. 0 to OFF the display, 1 to display cropped face, 2 to display head pose, 3 to display cropped eye, -1 to show main frame with final output, default option")
    return parser

def main():
    # read parameters from command line
    args = build_argparser().parse_args()
    # test argparser
    print(args.face)
    print(args.head)
    print(args.landmark)
    print(args.gaze)
    print(args.source)
    print(args.file)
    print(args.device)
    print(args.extension)
    print(args.results)

if __name__=='__main__':
    main()