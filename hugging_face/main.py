from huggingface_hub import HfApi, login

login(token="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")

api = HfApi()

repo_url = api.create_repo(
    repo_id="Bedimand/fraud-detector-v1",
    repo_type="model",
    exist_ok=True,
    private=False
)
print(f"Repositório: {repo_url}")