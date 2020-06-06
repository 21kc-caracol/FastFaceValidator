import mtcnn
import PIL
import numpy as np

class Pmtcnn():
    """
    post processes mtcnn for aligned faces accurate detection
    
    Prequisits- make sure you installed the standard mtcnn, PIL and numpy
                (pip install mtcnn...)
    
    methods(see their docstring): transform_detect_aligned
                                  change_scale_and_image_size
            
    
    """
    def __init__(self,scale_factor=0.709):
        self.model = mtcnn.MTCNN(scale_factor=scale_factor)
        self.dest_size= 50  # the target image size

    def transform_detect_aligned(self,image):
        """
        @param: image expects an image after it was loaded
        example: 
                image = PIL.Image.open(neg_image_path)
                
        transforms image to destination size and detects aligned faces
        
        Returns:
                if 1 landmark exceeds image size- returns False
                if all landmarks within image size= return True
                
                also assumes if there is 1 non-aligned face- returns false
        """
        image= image.resize((self.dest_size, self.dest_size), resample=PIL.Image.LANCZOS)
        image = image.convert('RGB')
        pixels = np.asarray(image)
        
        results = self.model.detect_faces(pixels)
        if len(results)==0:
            return False # didn't detect
        
        # could iterate once..but fot now I'll keep safe 
        for result in results:
            for key, loc in result['keypoints'].items():
                # just to be safe, adding "below zero conditions"
                if loc[0]>image.size[0] or loc[1]>image.size[1] or loc[0]<0 or loc[1]<0:
                    return False   
        
        return True
    
    def change_scale_and_image_size(self,scale_factor=0.709,dest_size=50 ):
        self.model = mtcnn.MTCNN(scale_factor=scale_factor)
        self.dest_size= dest_size  # the target image size

