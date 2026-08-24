import os
import torch
from modly import Generator   # interface provided by Modly
from PIL import Image

try:
    from hy3dgen.shape_v2 import Hunyuan3DStructurePipeline
except ImportError:
    Hunyuan3DStructurePipeline = None

class L3DGenerator(Generator):

    def load(self, variant):
        model_path = variant.get("path")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if Hunyuan3DStructurePipeline is not None:
            self.pipeline = Hunyuan3DStructurePipeline.from_pretrained(
                model_path, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            self.pipeline.to(device)
        else:
            self.pipeline = None

    def generate(self, image, slat_steps=5):
        if self.pipeline is None:
            raise RuntimeError("Hunyuan3D pipeline is niet correct geladen. Controleer je requirements.")

        input_image = image.convert("RGB")
        
        mesh_result = self.pipeline(input_image, num_inference_steps=slat_steps)
        
        output_path = os.path.join(os.getcwd(), "textured_output.glb")
        
        mesh_result.export(output_path)
            
        return output_path
