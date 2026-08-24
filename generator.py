import os
from modly import Generator   # interface provided by Modly

class MyGenerator(Generator):

    def load(self, variant):
        # Laad de gewichten van de geselecteerde variant (Base of Fast) op de GPU
        # Modly regelt de download via de URL uit de manifest.json
        self.model = load_weights(variant["url"])

    def generate(self, image):
        # image (PIL) -> AI pipeline genereert mesh + textuur
        # Zorg dat de run() methode van je model een getextureerde mesh (.glb) oplevert
        mesh_result = self.model.run(image)
        
        # Bepaal het bestandspad om de getextureerde GLB lokaal op te slaan
        output_path = os.path.join(os.getcwd(), "textured_output.glb")
        
        # Exporteer de mesh inclusief textuur naar het bestandspad
        # (Dit is afhankelijk van of je model direct een bestandspad, trimesh-object of bytes teruggeeft)
        if hasattr(mesh_result, 'export'):
            mesh_result.export(output_path)
        elif isinstance(mesh_result, str) and os.path.exists(mesh_result):
            output_path = mesh_result
            
        # Modly verwacht het lokale bestandspad naar de getextureerde mesh terug
        return output_path
