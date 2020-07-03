from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np

class HeadPose:
    '''
    Class for the Head Pose Estimation Model.
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

        self.output_name = list(self.network.outputs) # multiple outputs

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
        This method runs predictions on the input image.
        return: 1x3 array containing yaw, pitch, roll
        '''
        # Get processed data
        processed_image = self.preprocess_input(image)

        # start async request
        self.exec_network.start_async(0, inputs={self.input_name: processed_image})

        # get async request output
        status = self.exec_network.requests[0].wait(-1)
        if status == 0:
            outputs = []
            outputs.append(self.exec_network.requests[0].outputs[self.output_name[0]])
            outputs.append(self.exec_network.requests[0].outputs[self.output_name[1]])
            outputs.append(self.exec_network.requests[0].outputs[self.output_name[2]])

        # process output
        pose = self.preprocess_output(outputs)
        return pose

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

    def preprocess_output(self, outputs):
        '''
        Process the output to prepare for the next model's input
        return: 1x3 array containing yaw, pitch, roll values
        '''
        # extract the values
        yaw = float(outputs[0][0,0])
        pitch = float(outputs[1][0,0])
        roll = float(outputs[2][0,0])

        # create array
        pose = np.array([yaw, pitch, roll]).reshape(1,3)

        return pose

