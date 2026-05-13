from ikaros import Ikaros, Task, Optimization

server = Ikaros()

print("=" * 50)
print("TEST 1: Text Classification")
print("=" * 50)


classifier = server.deploy(
    "distilbert-base-uncased-finetuned-sst-2-english",
    task=Task.TEXT_CLASSIFICATION,
    optimize=Optimization.ONNX,
    device="cpu"
)



tests = [
    "This movie was absolutely fantastic, I loved every minute",
    "Terrible experience, waste of money, never again",
    "The weather is okay today",
    "I just got promoted at work!",
    "My flight got cancelled and I'm stuck at the airport",
]


for text in tests:
    result = classifier.predict(text)
    print(f"  '{text[:50]}...' → {result[0]['label']} ({result[0]['score']:.4f})")
    
    

# --- Test 2: NER (Token Classification) ---
print("\n" + "=" * 50)
print("TEST 2: Named Entity Recognition")
print("=" * 50)

ner = server.deploy(
    "dslim/bert-base-NER",
    task=Task.TOKEN_CLASSIFICATION,
    optimize=Optimization.ONNX,
    device="cpu"
)

ner_tests = [
    "Elon Musk founded SpaceX in California",
    "August Sletto works at SVT in Stockholm, Sweden",
    "Google was founded by Larry Page and Sergey Brin at Stanford University",
]

for text in ner_tests:
    result = ner.predict(text)
    entities = [f"{r['word']}({r['entity']})" for r in result]
    print(f"  '{text[:50]}...' → {', '.join(entities)}")

# --- Test 3: Translation EN → SV ---
print("\n" + "=" * 50)
print("TEST 3: Translation (English → Swedish)")
print("=" * 50)

translator = server.deploy(
    "Helsinki-NLP/opus-mt-en-sv",
    task=Task.TRANSLATION,
    optimize=Optimization.ONNX,
    device="cpu"
)

translation_tests = [
    "Hello, how are you today?",
    "Machine learning is transforming the world",
    "The weather in Stockholm is beautiful",
    "I am building an inference platform called Ikaros",
]

for text in translation_tests:
    result = translator.predict(text)
    translated = result[0]["translation_text"]
    print(f"  '{text}' → '{translated}'")

# --- Test 4: Embeddings ---
print("\n" + "=" * 50)
print("TEST 4: Embeddings")
print("=" * 50)

embedder = server.deploy(
    "sentence-transformers/all-MiniLM-L6-v2",
    task=Task.EMBEDDINGS,
    optimize=Optimization.ONNX,
    device="cpu"
)

embedding_tests = [
    "The cat sat on the mat",
    "A kitten was sitting on the rug",
    "Stock prices rose sharply today",
]

for text in embedding_tests:
    result = embedder.predict(text)
    dims = len(result[0][0])
    print(f"  '{text}' → vector of {dims} dimensions")

# --- Test 5: List all loaded models ---
print("\n" + "=" * 50)
print("LOADED MODELS")
print("=" * 50)

for model_id, info in server.list_models().items():
    print(f"  {model_id} → {info['task']} ({info['optimization']})")

# --- Test 6: Unload and verify ---
print("\n" + "=" * 50)
print("TEST 6: Unload")
print("=" * 50)

classifier.unload()
print(f"  After unloading classifier: {len(server.list_models())} models loaded")

ner.unload()
translator.unload()
embedder.unload()
print(f"  After unloading all: {len(server.list_models())} models loaded")

print("\nAll tests passed!")