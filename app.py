from flask import Flask, request, jsonify
# import summarizer

app = Flask(__name__)

# summarizer.initialize()

@app.route("/")
def index():
  return "<p>Hello World!</p>"

@app.route("/summarize", methods=['POST'])
def summarize():
  content = request.json
  article = content["article"]
  # summary = summarizer.greedy_decode(article, summarizer.model)
  return jsonify({"article": article, "summary": "tbd"})

# Test this:
# curl localhost:5000/summarize -H "Content-Type:application/json" -d '{"article": "adfasdf"}'
