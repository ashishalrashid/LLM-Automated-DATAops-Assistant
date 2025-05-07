# LLM Automated DATAops Assistant

**LLM Automated DATAops Assistant** is a Python-based automation framework that leverages Large Language Models (LLMs) to perform dynamic data handling, processing tasks, and on-the-fly code generation. Designed for extensibility and rapid prototyping, it exposes a REST API for seamless integration, powerful function-calling endpoints, and comes fully Dockerized for immediate deployment.

---

## 🚀 Core Functionality

* **Task Automation:** Execute tasks like file sorting, data extraction, code execution, and SQL queries with a single API call.
* **Automatic Code Generation:** For undefined tasks, the framework generates, validates, and executes Python code at runtime using LLMs.
* **Function Calling Interface:** Predefined endpoints map directly to core functions, allowing precise control over:

  * JSON sorting (`sort_json_file`)
  * Directory file indexing (`write_recent_file_lines`)
  * Text extraction via LLM (`llm_text_extractor`)
  * Embedding-based similarity search (`llm_find_similar_comments_using_embedding_model`)
  * SQL query execution (`execute_sql_query`)
  * Dynamic code execution (`execute_generated_code`)
* **Data Handling:** Supports JSON, CSV, text, and SQLite; input/output lives in a mounted `data/` volume.
* **Web Scraping & Shell Automation:** Capable of scraping websites and executing bash commands through LLM-generated prompts.

---

## 🔧 Major API Endpoints & Function Calling

| Endpoint   | Method | Query Parameters                                   | Description                                     |
| ---------- | ------ | -------------------------------------------------- | ----------------------------------------------- |
| `/run`     | POST   | `task` (string): natural language task description | Generates or dispatches code to handle the task |
| `/read`    | GET    | `path` (string): file path                         | Reads and returns file contents                 |
| `/sort`    | POST   | `input`, `output`, `keys`                          | Calls `sort_json_file`                          |
| `/extract` | POST   | `input`, `output`, `prompt`                        | Calls `llm_text_extractor`                      |
| `/sql`     | POST   | `db_file`, `query`, `output`                       | Calls `execute_sql_query`                       |

> **Automatic Dispatch:** If `/run` receives an unrecognized `task`, it uses the LLM to generate a function body, writes it to a temporary module, and calls `execute_generated_code`.

---

## 🤖 Automatic Code Generation Workflow

1. **Task Parsing:** `process_task(task: str)` parses the natural language description.
2. **LLM Prompting:** If no matching function exists, constructs a code-generation prompt.
3. **Code Synthesis:** LLM generates Python code implementing the task.
4. **Validation:** Runs static checks (e.g., syntax, safety patterns).
5. **Execution:** Calls `execute_generated_code(code, args)` to run and return outputs.
6. **Cleanup:** Removes temporary modules and returns results via API response.

---

## 🐳 Dockerized Deployment

All dependencies and environment variables are encapsulated in Docker for consistent runtime.

```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t llm-automation-master .
docker run -d -p 8000:8000 -v $(pwd)/data:/app/data --env-file .env llm-automation-master
```

---

## ⚙️ Extensibility

* **Add New Functions:** Define your Python function in `app.py` or a plugin module and register in the task router.
* **Custom Tools:** Integrate external APIs or CLI tools by wrapping calls in new function endpoints.
* **LLM Fine-Tuning:** Swap the underlying LLM or adjust prompts via configuration.

---

## 📦 Dependencies

The project uses the following Python libraries:

* `fastapi`: For building the REST API.
* `uvicorn`: ASGI server for running the FastAPI app.
* `requests`: For making HTTP requests.
* `pandas`: Data manipulation and analysis.
* `python-dotenv`: For managing environment variables.
* `regex`: Regular expression operations.
* `python-dateutil`: Date parsing and manipulation.
* `numpy`: Numerical operations.
* `sqlite3`: SQLite database operations.

Refer to `requirements.txt` for the full list.

---

## 📬 Contact & Contribution

For issues, feature requests, or pull requests, please open an issue or PR on the GitHub repo.

**Author:** Ashish Al Rashid 

**Repository:** [https://github.com/your-username/LLM-Automation-Master](https://github.com/your-username/LLM-Automation-Master)
