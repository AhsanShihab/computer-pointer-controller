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
        return: Bounding box coordinates for detected face
        '''
        # store original height, width and channel
        oh, ow, c = image.shape

        # Get processed data
        processed_image = self.preprocess_input(image)

        # start async request
        self.exec_network.start_async(0, inputs={self.input_name: processed_image})

        # get async request output
        status = self.exec_network.requests[0].wait(-1)
        if status == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_name]
        
        # return face bounding box
        xmin, ymin, xmax, ymax = self.preprocess_output(outputs)
        xmin, ymin, xmax, ymax = int(xmin*ow), int(ymin*oh), int(xmax*ow), int(ymax*oh)

        return xmin, ymin, xmax, ymax

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
        Extracts the coordinates of the detected face with highest confidence
        return: Touple containing bounding box coordinates
        '''
        boxes = outputs[0][0]
        max_conf = 0.4              # threshold value
        max_box = 0
        face_found = False

        # find the location of the face with highest confidence
        for i in range(len(boxes)):
            box = boxes[i]
            conf = box[2]
            if conf == max(conf, max_conf):
                face_found = True
                max_conf = conf
                max_box = i
        
        # get bounding box coordinates of the face
        if face_found:
            box = boxes[max_box]
            xmin = box[3]
            ymin = box[4]
            xmax = box[5]
            ymax = box[6]
        
            return xmin, ymin, xmax, ymax
        
        return 0, 0, 0, 0
