import os
from modly import Generator

class MyGenerator(Generator):
    
    def load(self, variant):
        self.model = load_weights(variant["url"])
        
    def generate(self, image):
        mesh_result = self.model.run(image)
        output_path = os.path.join(os.getcwd(), "textured_output.glb")
        if hasattr(mesh_result, 'export'):
            mesh_result.export(output_path)
        elif isinstance(mesh_result, str) and os.path.exists(mesh_result):
            output_path = mesh_result
        return output_path
