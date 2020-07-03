from openvino.inference_engine import IENetwork, IECore
import cv2

class FaceDetection:
    '''
    Class for the Face Detection Model.
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

        # Adding extension if needed
        if extension_needed:
            print(self.device + " extension needed. Adding " + self.device + " extension")
            try:
                self.core.add_extension(extension_path=self.ext, device_name=self.device)
            except Exception as e:
                print("Couldn't add extension. Privide correct extension file for " + self.device + ".")
                raise
            
            # Recheck if all layers are now supported after adding extension
            layer_map = self.core.query_network(network=self.network, device_name=self.device)
            supported_layers = layer_map.keys()
            extension_needed = False

            for layer in network_layers:
                if layer in supported_layers:
                    pass
                else:
                    extension_needed=True
            
            if extension_needed:
                print('Some layers are unsupported. Exit program.')
                exit(1)
            else:
                print("Successfully loaded extension.")
        
        # Loading model
        self.exec_network = self.core.load_network(self.network, self.device)

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Preprocess the input image to the expected input shape for the model
        output: preprocessed image
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
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError
