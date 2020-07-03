from openvino.inference_engine import IENetwork, IECore
import cv2

class FacialLandmarks:
    '''
    Class for the Facial Landmarks Detection Model.
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

        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape

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

    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        return: Coordinates of left eye and right eye
        '''
        # Get processed data
        processed_image = self.preprocess_input(image)

        # start async request
        self.exec_network.start_async(0, inputs={self.input_name: processed_image})

        # get async request output
        status = self.exec_network.requests[0].wait(-1)
        if status == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_name]
        
        # Get the eye locations
        left_eye, right_eye = self.preprocess_output(outputs, image)

        return left_eye, right_eye

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

    def preprocess_input(self, image):
        '''
        Preprocess the input image to the expected input shape for the model
        return: preprocessed image
        '''
        # get model's expected shape
        batch, channel, height, width = self.input_shape

        # processs image to match input
        image = cv2.resize(image, (width, height))
        image = image.transpose((2, 0, 1)) # bringing the color channel first
        image = image.reshape(1, 3, height, width) # adds batch size to match model's expected input shape

        return image

    def preprocess_output(self, outputs, image):
        '''
        Process the output and extracts the bounding box coordinates of left eye and right eye
        return: Two touples containing two eyes (left eye followed by right eye) bouding box coordinates
        '''
        # original height, width and channel
        oh, ow, c = image.shape

        # extract left eye and right eye centers' coordinates
        left_eye_x = outputs[0,0]
        left_eye_y = outputs[0,1]
        right_eye_x = outputs[0,2]
        right_eye_y = outputs[0,3]

        # convert coordinates with respect to main image
        left_eye_x = int(left_eye_x * ow)
        left_eye_y = int(left_eye_y * oh)
        right_eye_x = int(right_eye_x * ow)
        right_eye_y = int(right_eye_y * oh)

        # box around left eye, 30px space on each side from the center, resulting 60x60px box
        left_eye_xmin, left_eye_xmax = max(left_eye_x - 30, 0), min(left_eye_x + 30, ow)
        left_eye_ymin, left_eye_ymax = max(left_eye_y - 30, 0), min(left_eye_y + 30, oh)

        # box around right eye, 30px space on each side from the center, resulting 60x60px box
        right_eye_xmin, right_eye_xmax = max(right_eye_x - 30, 0), min(right_eye_x + 30, ow)
        right_eye_ymin, right_eye_ymax = max(right_eye_y - 30, 0), min(right_eye_y + 30, oh)

        # pack the result in two touples
        left_eye = (left_eye_xmin, left_eye_ymin, left_eye_xmax, left_eye_ymax)
        right_eye = (right_eye_xmin, right_eye_ymin, right_eye_xmax, right_eye_ymax)

        return left_eye, right_eye
