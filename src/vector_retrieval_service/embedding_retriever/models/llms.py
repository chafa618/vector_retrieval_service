from sentence_transformers import SentenceTransformer, util # type: ignore
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Clase que usa un modelo SentenceTransformer para vectorizar un texto y obtener embeddings
    """
    def __init__(self, model_name):
        logger.info(f"Cargando modelo de embeddings {model_name}...")
        try:
            self.model = SentenceTransformer(model_name, token=os.environ.get("HF_TOKEN"))
            logger.info("Modelo de embeddings cargado exitosamente.")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo de embeddings: {e}")
            raise e

    def get_embedding(self, text):
        """
        Devuelve el embedding de un texto.
        Args:
            text (str): Texto a procesar.
        Returns:
            np.array: Vector con el embedding.
        """
        return self.model.encode(text)





if __name__ == "__main__":
    vectorizer = EmbeddingModel()
    query = "Cuantos habitantes tiene Londres?"
    docs = ["En Londres viven alrededor de 9000 personas. ", "Londres es reconocido como un distrito financiero."]

    query_emb = vectorizer.get_embedding(query)
    doc_emb = vectorizer.get_embedding(docs)

    #Compute dot score between query and all document embeddings
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    #Combine docs & scores
    doc_score_pairs = list(zip(docs, scores))

    #Sort by decreasing score
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    #Output passages & scores
    for doc, score in doc_score_pairs:
        print(score, doc)