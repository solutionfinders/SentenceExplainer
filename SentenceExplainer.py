import random
import string

import numpy as np
import pandas as pd
from IPython.core.display import HTML
from IPython.core.display import display as ipython_display
from matplotlib.colors import LinearSegmentedColormap
from sentence_transformers import util


# An Explainer for Sentence Transformers
class SentenceExplainer:
    def __init__(self, model, corpus_embeddings, papers):
        self.model = model
        self.corpus_embeddings = corpus_embeddings
        self.papers = papers

    def search_and_explain(self, sentence):
        # search for similar papers, and explain the results
        # return a dataframe with the results and explanations

        sentence_embedding = self.model.encode(sentence, convert_to_tensor=True)

        search_hits = util.semantic_search(sentence_embedding, self.corpus_embeddings, top_k=10) # return the top 10 papers
        search_hits = search_hits[0]  # Get the 10 hits for the "first" (and only) query

        # resulting dataframe
        # for every word in the sentence one column with the delta score
        results = pd.DataFrame(
            columns=["paper_id", "score", "title"] + sentence.split()
        )

        hit_ids = []
        hit_scores = {}
        for hit in search_hits:
            hit_ids = hit_ids + [hit["corpus_id"]]
            related_paper = self.papers[hit["corpus_id"]]
            hit_scores[hit["corpus_id"]] = hit["score"]
            new_row = pd.Series(
                {
                    "paper_id": hit["corpus_id"],
                    "score": hit["score"],
                    "title": related_paper["title"],
                },
                name=hit["corpus_id"],
            )

            results.loc[len(results.index)] = new_row

        all_index_text = range(0, len(sentence.split()))
        for part_of_text in all_index_text:
            shorter_index = sentence.split()
            popped_word = shorter_index.pop(
                part_of_text
            )  # delete from the split strings the one we do not want to see now
            shorter_text = " ".join(shorter_index)

            shorter_embedding = self.model.encode(
                shorter_text, convert_to_tensor=True
            )  # encode the shorter text
            search_hits_popped_word = util.semantic_search(
                shorter_embedding, self.corpus_embeddings[hit_ids]
            )[0]# now get the new scores only for the 10 papers we have already found for the whole sentence
            search_hits_popped_word = search_hits_popped_word  # Get the hits for the first query

            for hit_popped_word in search_hits_popped_word:
                # the original corpus id is the new corpus id translated by the hit_ids from above
                original_corpus_id = hit_ids[
                    hit_popped_word["corpus_id"]
                ]  
                related_paper = self.papers[original_corpus_id]
                # the difference this one deleted word makes on the prediction score of this paper:
                delta_score = (
                    hit_scores[hit_ids[hit_popped_word["corpus_id"]]]
                    - hit_popped_word["score"]
                )
                results.loc[
                    results.paper_id == original_corpus_id, popped_word
                ] = delta_score

        return results

    def mark_text(self, result_df, paper_id):
        # a function to mark text from red to blue depending on the score

        uuid = "".join(random.choices(string.ascii_lowercase, k=20))
        out = f""
        output_values = result_df[result_df.paper_id == paper_id].values[0, 3:]
        output_max = np.max(np.abs(output_values))

        # inspired by shap
        colors = []
        for l in np.linspace(1, 0, 100):
            colors.append((30.0 / 255, 136.0 / 255, 229.0 / 255, l))
        for l in np.linspace(0, 1, 100):
            colors.append((255.0 / 255, 13.0 / 255, 87.0 / 255, l))
        red_transparent_blue = LinearSegmentedColormap.from_list(
            "red_transparent_blue", colors
        )

        for i, name in enumerate(result_df.columns[3:]):
            scaled_value = 0.5 + 0.5 * output_values[i] / (output_max + 1e-8)
            color = red_transparent_blue(scaled_value)
            color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])
            out += f"""
        <div style="display: inline; background: rgba{color}; border-radius: 3px; padding: 0px" id="_tp_{uuid}_output_{i}_name">{name}</div>"""
        out += "</div>"
        ipython_display(HTML(out))

    def explain(self, sentence):
        # final function, combine it all and print the results with marked text
        result_df = self.search_and_explain(sentence)
        for paper in result_df.paper_id:
            print(
                "{} \t{:.2f}\t{} \n\n{}".format(
                    paper,
                    result_df[result_df.paper_id == paper].score.values[0],
                    result_df[result_df.paper_id == paper].title.values[0].strip('""'),
                    self.papers[paper]["abstract"].strip('""')[:300] + "...",
                )
            )
            self.mark_text(result_df, paper_id=paper)
