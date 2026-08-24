import os
from modly import Generator

class L3DGenerator(Generator):

    def load(self, variant):
        pass

    def generate(self, image, slat_steps=25):
        output_path = os.path.join(os.getcwd(), "textured_output.glb")
        return output_path
