from locust import HttpUser, task, between

class IkarosUser(HttpUser):
    wait_time = betweem(0.01, 0.05)
    
    @task
    def predict(self):
        self.client.post("/predict", json={
            "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
            "input": "This product is absolutely wonderful and I love it"
        })