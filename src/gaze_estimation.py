from openvino.inference_engine import IENetwork, IECore
import cv2
import math

class Gaze:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'
        self.device = device
        self.ext = extensions

        self.core = IECore()
        try:
            try:
                # for openVINO version 2020.3
                self.network = self.core.read_network(model=self.model_structure, weights=self.model_weights)
            except AttributeError:
                # for openVINO version 2019
                self.network = IENetwork(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise

        self.input_name = (list(self.network.inputs))
        #['head_pose_angles', 'left_eye_image', 'right_eye_image']

        self.output_name = next(iter(self.network.outputs))

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        return: None
        '''
        # check for unsupported layers
        extension_needed = self.check_model()

        # Adding extension if needed
        if extension_needed:
            print(self.device + " extension needed. Adding " + self.device + " extension")
            try:
                self.core.add_extension(extension_path=self.ext, device_name=self.device)
            except Exception as e:
                print("Couldn't add extension. Privide correct extension file for " + self.device + ".")
                raise
            
            # Recheck if all layers are now supported after adding extension
            extension_needed = self.check_model()
            
            if extension_needed:
                print('Some layers are unsupported. Exit program.')
                exit(1)
            else:
                print("Successfully loaded extension.")
        
        # Loading model
        self.exec_network = self.core.load_network(self.network, self.device)

    def predict(self, head_pose_angles, left_eye_image, right_eye_image):
        '''
        This method is meant for running predictions on the input data.
        return: x and y of gaze vector
        '''
        # Get processed data
        left_eye_image, right_eye_image = self.preprocess_input(left_eye_image, right_eye_image)

        # set up the input dictionary
        inputs = {self.input_name[0]: head_pose_angles, 
                self.input_name[1]: left_eye_image, 
                self.input_name[2]: right_eye_image}

        # start async request
        self.exec_network.start_async(0, inputs=inputs)

        # get async request output
        status = self.exec_network.requests[0].wait(-1)
        if status == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_name]
        
        # normalize the gaze vectors
        outputs = outputs / cv2.norm(outputs)

        # extract x, y and roll
        x = outputs[0,0]
        y = outputs[0,1]
        roll = head_pose_angles[0,2]

        # process output
        outputs = self.preprocess_output(roll, x, y)

        return outputs

    def check_model(self):
        """
        Checks the model if there is any unsupported layers
        return: True if unsupported layer exists, False otherwise
        """
        network_layers = self.network.layers.keys()
        layer_map = self.core.query_network(network=self.network, device_name=self.device)
        supported_layers = layer_map.keys()
        extension_needed = False

        for layer in network_layers:
            if layer in supported_layers:
                pass
            else:
                extension_needed=True
                break
        
        return extension_needed

    def preprocess_input(self, left_eye_image, right_eye_image):
        '''
        Preprocess the input image to the expected input shape for the model
        return: preprocessed left eye and right eye images
        '''
        # get model input shape
        left_eye_shape = self.network.inputs[self.input_name[1]].shape
        right_eye_shape = self.network.inputs[self.input_name[2]].shape

        # get expected height and width
        left_height, left_width = left_eye_shape[2], left_eye_shape[3]
        right_height, right_width = right_eye_shape[2], right_eye_shape[3]

        # process left eye image
        left_eye_image = cv2.resize(left_eye_image, (left_width, left_height))
        left_eye_image = left_eye_image.transpose((2, 0, 1)) # bringing the color channel first
        left_eye_image = left_eye_image.reshape(1, 3, left_height, left_width) # adding batch size to match model's expected input shape

        # process right eye image
        right_eye_image = cv2.resize(right_eye_image, (right_width, right_height))
        right_eye_image = right_eye_image.transpose((2, 0, 1)) # bringing the color channel first
        right_eye_image = right_eye_image.reshape(1, 3, right_height, right_width) # adding batch size to match model's expected input shape
        
        return left_eye_image, right_eye_image

    def preprocess_output(self, roll, x, y):
        '''
        Modifies x and y according to head position
        return: x and y
        '''
        # rotate gaze vector to compensate for the head alignment
        cs = math.cos(roll * math.pi / 180)
        sn = math.sin(roll * math.pi / 180)

        x = x * cs + y * sn
        y = -x * sn + y * cs

        return x, y
