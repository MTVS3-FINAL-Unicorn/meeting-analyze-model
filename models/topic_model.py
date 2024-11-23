import pyLDAvis
import pyLDAvis.lda_model
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV


class TopicModel():
    def __init__(self, token_list):
        self.count_vec = CountVectorizer(max_df=10, max_features=1000, min_df=1, ngram_range=(1,2))
        self.feat_vec = self.count_vec.fit_transform(token_list)
        self.lda = LatentDirichletAllocation(random_state=42)
        self.param_grid = {'n_components': [3, 4, 5]}
        self.search = GridSearchCV(self.lda, self.param_grid, cv=3)
        self.search.fit(self.feat_vec)
        self.best_model = self.search.best_estimator_
        self.feature_names = self.count_vec.get_feature_names_out()

    def show_topics(self, num_top_words):
        for topic_idx, topic in enumerate(self.best_model.components_):
            print(f'Topic {topic_idx+1}')
            
            topic_word_idx = topic.argsort()[::-1]
            top_idx = topic_word_idx[:num_top_words]
            
            result_text = ' '.join([self.feature_names[i] for i in top_idx])
            print(result_text)
    
    def make_html(self):
        try:
            # Prepare the visualization data
            vis = pyLDAvis.lda_model.prepare(self.best_model, self.feat_vec, self.count_vec)
            vis_file = f'{self.meeting_id}.html'
            pyLDAvis.save_html(vis, './'+vis_file)
            return vis_file
        except Exception as e:
            print(f"An error occurred while creating the visualization: {str(e)}")
    
    def prepared_html_data(self, vis_data):
        # 변환된 JSON 데이터 생성
        html_data = {
            "mdsDat": {
                "x": vis_data.topic_coordinates['x'].tolist(),
                "y": vis_data.topic_coordinates['y'].tolist(),
                "topics": vis_data.topic_coordinates['topics'].tolist(),
                "cluster": vis_data.topic_coordinates['cluster'].tolist(),
                "Freq": vis_data.topic_coordinates['Freq'].tolist(),
            },
            "tinfo": {
                "Term": vis_data.topic_info['Term'].tolist(),
                "Freq": vis_data.topic_info['Freq'].tolist(),
                "Total": vis_data.topic_info['Total'].tolist(),
                "Category": vis_data.topic_info['Category'].tolist(),
                "logprob": vis_data.topic_info['logprob'].tolist(),
                "loglift": vis_data.topic_info['loglift'].tolist(),
            },
            "token.table": {
                "Topic": vis_data.token_table['Topic'].tolist(),
                "Freq": vis_data.token_table['Freq'].tolist(),
                "Term": vis_data.token_table['Term'].tolist(),
            },
            "R": vis_data.R,
            "lambda.step": vis_data.lambda_step,
            "plot.opts": vis_data.plot_opts,
            "topic.order": vis_data.topic_order
        }
        
        return html_data

    def make_lda_json(self):
        vis_data = pyLDAvis.lda_model.prepare(self.best_model, self.feat_vec, self.count_vec)
        json_data = self.prepared_html_data(vis_data)
        return json_data

if __name__ == '__main__':
    token_list = ['분홍색', '테마', '시즌', '정말', '부드러운', '이미지', '같아요', '고객', '좋아할', '같네요', '개인', '초록색', '브랜드', '정체', '있다고', '생각', '분홍색', '흔한', '느낌', '같아요', '차라리', '이번', '초록색', '어떨까', '초록색', '자연', '이미지', '강해서', '요즘', '트렌드', '같아요', '분홍색', '상큼', '여성', '이미지', '젊은', '인기', '많을', '초록색', '테마', '시원하고', '깨끗한', '이미지', '있을', '같아요', '분홍색', '정말', '제품', '꽃잎', '성분', '생각', '브랜드', '이미지', '관성', '유지', '초록색', '적합할', '같습니다', '분홍색', '화장품', '패키지', '같아요', '초록색', '환경', '메시지', '강화할', '있어서', '좋을', '같아요', '분홍색', '화사한', '느낌', '부담', '사용', '있을', '같아요', '초록색', '브랜드', '생각', '자연', '관련', '브랜드', '라면', '분홍색', '따뜻하고', '부드러운', '이미지', '주기', '때문', '소비자', '긍정', '반응', '있을', '같아요', '초록색', '자연스러운', '이미지', '강화하는', '좋을', '같아요', '친환경', '느낌', '강해요', '분홍색', '꽃잎', '영감', '만큼', '제품', '컬러', '생각', '초록색', '시각', '강렬한', '인상', '있어서', '좋을', '같아요', '분홍색', '고급스러운', '느낌', '있다고', '생각', '초록색', '브랜드', '자연', '이미지', '더욱', '있을', '같습니다', '분홍색', '소비자', '편안한', '느낌', '있어서', '좋다고', '생각', '초록색', '세련된', '이미지', '있을', '같아요']
    topic_model = TopicModel(token_list, 0, 0)
    topic_model.make_html()
    topic_model.make_lda_json()