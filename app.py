import cv2 
from argparse import ArgumentParser
from src.input_feeder import InputFeeder
from src.mouse_controller import MouseController
from src.face_detection import FaceDetection
from src.head_pose_estimation import HeadPose
from src.facial_landmarks_detection import FacialLandmarks
from src.gaze_estimation import Gaze


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

def crop_image(boundary, image):
    """
    Crops the given image based on the given boundary coordinates
    return: Cropped Image
    """
    # get the coordinates
    xmin, ymin, xmax, ymax = boundary
    # crop the image
    cropped_image = image[ymin:ymax, xmin:xmax]

    return cropped_image

def main():
    # read parameters from command line
    args = build_argparser().parse_args()
    
    # Load models
    print(("Loading Face Detection Model."))
    face = FaceDetection(model_name=args.face, device=args.device, extensions=args.extension)
    face.load_model()
    print("Face detection model loaded successfully.\n")

    print(("Loading Head Pose Estimation Model."))
    head_pose = HeadPose(model_name=args.head, device=args.device, extensions=args.extension)
    head_pose.load_model()
    print("Head Pose Estimation model loaded successfully.\n")

    print(("Facial Landmarks Detection Model."))
    facial_landmark = FacialLandmarks(model_name=args.landmark, device=args.device, extensions=args.extension)
    facial_landmark.load_model()
    print("Facial Landmarks Detection model loaded successfully.\n")

    print(("Gaze Estimation Model."))
    gaze = Gaze(model_name=args.gaze, device=args.device, extensions=args.extension)
    gaze.load_model()
    print("Gaze Estimation model loaded successfully.\n")

    print(("All model loaded successfully"))

    # Create Mouse controller object
    mouse = MouseController(precision="high", speed="fast")

    # Read video feed
    feed = InputFeeder(input_type=args.source, input_file=args.file)
    feed.load_data()

    # process video feed
    for frame in feed.next_batch():
        # detect face
        face_found, face_location = face.predict(frame)
        if face_found:
            # prepare for proceeding to the next models
            cropped_face = crop_image(face_location, frame)

            # get head pose
            head_position = head_pose.predict(frame)

        else:
            pass

        if args.results == 1:
            frame = cv2.rectangle(frame, (face_location[0], face_location[1]), (face_location[2], face_location[3]), (0,255,0), 2) 
            frame = cv2.resize(frame, (800, 400))
            cv2.imshow("Frame", frame)

        elif args.results == 2:
            frame = cv2.resize(frame, (800, 400))
            yaw, pitch, roll = head_position[0, 0], head_position[0, 1], head_position[0, 2]
            cv2.putText(frame, "yaw: " + str(yaw), (5,10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "pitch: " + str(pitch), (5,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "roll: " + str(roll), (5,50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow("Head Position", frame)
        

        # listening for key press to break, press q to break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            logging.info("Break key pressed") 
            break

    

if __name__=='__main__':
    main()