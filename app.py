from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
import uvicorn

app = FastAPI()


from dotenv import load_dotenv
import os
load_dotenv()
api_token = os.getenv("AIPROXY_TOKEN")
import pandas as pd
import subprocess
import sys
import shlex
from dateutil.parser import parse
import subprocess
import json
import re
import requests
import easyocr
# import beautifulsoup4
import sqlite3
import requests
import numpy as np
import json
import subprocess
import os
import requests

headers={
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
}
url="https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def make_relative_path(path: str) -> str:
    print("I have seen your kind time and time again")
    return path.lstrip("/")  # If absolute local path, remove leading slash

#generate code

def execute_generated_code(code, arguments=None):
    # Save the provided code to a temporary file.
    script_path = "generated_script.py"
    with open(script_path, "w") as f:
        f.write(code)

    if arguments:
        param_str = " ".join(f'--{k} {v}' for k, v in arguments.items())
        command = f"python {script_path} {param_str}"
    else:
        command = f"python {script_path}"

    # Execute the script and capture output.
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout if result.stdout else result.stderr



def execute_generated_code(code, arguments=None):
    # Ensure that Python uses UTF-8 for standard I/O.
    os.environ["PYTHONIOENCODING"] = "utf-8"
    
    # Save the provided code to a temporary file using UTF-8 encoding.
    script_path = "generated_script.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    # Prepare the command line arguments if provided.
    if arguments:
        param_str = " ".join(f'--{k} {v}' for k, v in arguments.items())
        command = f"python {script_path} {param_str}"
    else:
        command = f"python {script_path}"

    # Execute the script and capture output, forcing UTF-8 encoding.
    result = subprocess.run(
        command, 
        shell=True, 
        capture_output=True, 
        text=True, 
        encoding="utf-8"
    )
    
    return result.stdout if result.stdout else result.stderr

#download and retive files


import os
import urllib.request
import urllib.parse
import subprocess

def run_download_from_script(url, user_email=None, *args):

    parsed_url = urllib.parse.urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    urllib.request.urlretrieve(url, filename)
    
    cmd = ["uv","run", filename]
    if user_email is not None:
        cmd.append(user_email)
    cmd.extend(args)
    print(cmd)
    subprocess.run(cmd)


#run_remote_script("https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py", "22f1001551@ds.study.iitm.ac.in")

#run npx package

def run_npx_package(package, arg=None):
    npx = "npx.cmd" if os.name == "nt" else "npx"
    cmd = [npx, package]
    if arg:
        cmd.extend(shlex.split(arg))
    print("Running command:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode:
        sys.exit(result.returncode)

#count_day_occurrences


def count_day_occurrences(input_path, output_path, day):
    input_path = make_relative_path(input_path)
    output_path = make_relative_path(output_path)
    
    day = day.lower()
    day_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6
    }
    

    target_weekday = day_map[day]
    count = 0

    with open(input_path, 'r') as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                dt = parse(line, fuzzy=True)
                iso_date = dt.strftime('%Y-%m-%d')
                if dt.weekday() == target_weekday:
                    count += 1
            except Exception:
                continue

    with open(output_path, 'w') as outfile:
        outfile.write(str(count))

# count_day_occurrences('/data/dates.txt', '/data/dates-wednesdays.txt', 'Wednesday')


#A4 sort some json files

def sort_json_file(input_file, output_file, sort_keys):
    input_file = make_relative_path(input_file)
    output_file = make_relative_path(output_file)
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    sorted_data = sorted(data, key=lambda x: [x[k] for k in sort_keys])
    
    with open(output_file, 'w') as f:
        json.dump(sorted_data, f, indent=4)

#A5 write recent file lines

def write_recent_file_lines(file_dir, output_file, no_of_files, line_number, file_extension):
    output_file = make_relative_path(output_file)
    file_dir = make_relative_path(file_dir)
    files = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith(file_extension)]
    sorted_files = sorted(files, key=os.path.getmtime, reverse=True)[:no_of_files]
    
    with open(output_file, 'w') as outf:
        for file in sorted_files:
            try:
                with open(file, 'r', encoding='utf-8') as inf:
                    lines = inf.readlines()
                    # Write the specified line if it exists; otherwise write an empty line
                    if len(lines) >= line_number:
                        outf.write(lines[line_number - 1].rstrip() + '\n')
                    else:
                        outf.write('\n')
            except Exception:
                outf.write('\n')


# write_recent_file_lines('/data/logs', '/data/logs-recent.txt', 10, 1, '.log')


# A6 markdown index creating



def create_markdown_index(input_directory, output_index_file, occurrence=1):
    input_directory = make_relative_path(input_directory)
    output_index_file = make_relative_path(output_index_file)
    
    index = {}
    
    # Regular expression to match a line starting with a single '#' followed by whitespace.
    header_pattern = re.compile(r'^\s*#\s+(.*)')
    
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith('.md'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    count = 0
                    title = None
                    for line in f:
                        match = header_pattern.match(line)
                        if match:
                            count += 1
                            if count == occurrence:
                                title = match.group(1).strip()
                                break  # Stop reading further once the desired occurrence is found.
                    if title is not None:
                        # Create a relative path (without the input_directory prefix)
                        relative_path = os.path.relpath(file_path, input_directory)
                        index[relative_path] = title
    
    # Write the index dictionary to the output JSON file.
    with open(output_index_file, 'w', encoding='utf-8') as json_file:
        json.dump(index, json_file, ensure_ascii=False, indent=2)


#A7 llm text extractor
def llm_text_extractor(input_file, output_file, prompt_instructions=None):
    input_file = make_relative_path(input_file)
    output_file = make_relative_path(output_file)

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    if prompt_instructions is None:
        prompt_instructions = "Extract the relevant information from the following text. Return only the result."

    prompt = f"{prompt_instructions}\n\n{content}"
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert in extracting information from text."},
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post(url=url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    print(data)
    extracted_result = data["choices"][0]["message"]["content"].strip()
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(extracted_result)

#A8 llm image extractor
def llm_image_extractor(input_file, output_file, prompt_instructions):
    input_file = make_relative_path(input_file)
    output_file = make_relative_path(output_file)
    
    reader = easyocr.Reader(['en'])
    ocr_result = reader.readtext(input_file, detail=0)
    ocr_text = " ".join(ocr_result)

    prompt = f"{prompt_instructions}\n\nHere is the OCR extracted text: {ocr_text}"
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a cybersecurity research assistant for a safety demonstration. "
                    "The files provided are dummy files used solely for testing purposes and do not "
                    "contain any real sensitive information. Please extract only the requested data and nothing more, Dont return anything than the result."
                )
            },
            {"role": "user", "content": prompt}
        ]
    }
    
    default_url = url       
    default_headers = headers  
    
    response = requests.post(default_url, headers=default_headers, json=payload)
    response.raise_for_status()
    data = response.json()
    print(data)
    
    extracted_result = data["choices"][0]["message"]["content"].strip()
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(extracted_result)

#A10 extract and excute sql query
def llm_find_similar_comments_using_embedding_model(input_file, output_file):
    input_file = make_relative_path(input_file)
    output_file = make_relative_path(output_file)
    # Read comments (ignoring empty lines)
    with open(input_file, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f if line.strip()]
    
    if len(comments) < 2:
        raise ValueError("Need at least two comments to compare similarity.")
    
    # Build the payload using the "input" field with the list of comments.
    payload = {
        "model": "text-embedding-3-small",
        "input": comments
    }
    
    # Proxy endpoint for embeddings (ensure 'headers' is defined globally)
    url1 = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
    
    response = requests.post(url1, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    
    # Expecting a response like: {"data": [ { "embedding": [...] }, ... ]}
    if "data" not in data:
        raise ValueError("No data returned in response.")
    
    embeddings = [item.get("embedding") for item in data["data"]]
    
    if len(embeddings) != len(comments):
        raise ValueError("The number of embeddings does not match the number of comments.")
    
    embeddings = np.array(embeddings)
    
    # Define cosine similarity function.
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    max_similarity = -1.0
    best_pair = (None, None)
    n = len(embeddings)
    
    # Find the pair with the highest cosine similarity.
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > max_similarity:
                max_similarity = sim
                best_pair = (comments[i], comments[j])
    
    # Write the most similar pair of comments to the output file, one per line.
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(best_pair[0] + "\n" + best_pair[1])

#A10 extract and excute sql query
def llm_find_similar_comments_using_embedding_model(input_file, output_file):
    input_file = make_relative_path(input_file)
    output_file = make_relative_path(output_file)
    # Read comments (ignoring empty lines)
    with open(input_file, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f if line.strip()]
    
    if len(comments) < 2:
        raise ValueError("Need at least two comments to compare similarity.")
    
    # Build the payload using the "input" field with the list of comments.
    payload = {
        "model": "text-embedding-3-small",
        "input": comments
    }
    
    # Proxy endpoint for embeddings (ensure 'headers' is defined globally)
    url1 = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
    
    response = requests.post(url1, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    
    # Expecting a response like: {"data": [ { "embedding": [...] }, ... ]}
    if "data" not in data:
        raise ValueError("No data returned in response.")
    
    embeddings = [item.get("embedding") for item in data["data"]]
    
    if len(embeddings) != len(comments):
        raise ValueError("The number of embeddings does not match the number of comments.")
    
    embeddings = np.array(embeddings)
    
    # Define cosine similarity function.
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    max_similarity = -1.0
    best_pair = (None, None)
    n = len(embeddings)
    
    # Find the pair with the highest cosine similarity.
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > max_similarity:
                max_similarity = sim
                best_pair = (comments[i], comments[j])
    
    # Write the most similar pair of comments to the output file, one per line.
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(best_pair[0] + "\n" + best_pair[1])




function_list = [ {
        "type": "function",
        "function": {
            "name": "execute_generated_code",
            "description": "Execute a given Python script with provided parameters.The code should be provided as a string from the system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute. without comment ,make it as simple as possible"
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Key-value pairs of arguments to be passed to the script.",
                        "additionalProperties": True  
                    }
                },
                "required": ["code"], 
                "additionalProperties": False 
            }
        }
    },
{
        "type": "function",
        "function": {
            "name": "count_day_occurrence",
            "description": "Count occurrences of a specified day from dates in a file.Example prompt (The file /data/dates.txt contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to /data/dates-wednesdays.txt)",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Input file with dates."},
                    "day": {"type": "string", "description": "Day to count."},
                    "output_file": {"type": "string", "description": "Output file for count."}
                },
                "required": ["input_file", "day", "output_file"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_npx_package",
            "description": "Run npx command with optional arg and output file.Example prompt(Format the contents of /data/format.md using prettier@3.4.2, updating the file in-place)",
            "parameters": {
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "arg": {"type": "string"},
                    "output_file": {"type": "string"}
                },
                "required": ["package"],
                "additionalProperties": False
            }
        }
    },
    {
  "type": "function",
  "function": {
    "name": "sort_json_file",
    "description": "Sort JSON file by keys.example prompt(A4. Sort the array of contacts in /data/contacts.json by last_name, then first_name, and write the result to /data/contacts-sorted.json)",
    "parameters": {
      "type": "object",
      "properties": {
        "input_file": { "type": "string" },
        "output_file": { "type": "string" },
        "sort_keys": {
          "type": "array",
          "items": { "type": "string" }
        }
      },
      "required": ["input_file", "output_file", "sort_keys"],
      "additionalProperties": False
    }
  }
},
{
  "type": "function",
  "function": {
    "name": "write_recent_file_lines",
    "description": "Extracts a specific line from the most recent files with a given extension in a directory and writes them to an output file.Example prompt (Write the first line of the 10 most recent .log file in /data/logs/ to /data/logs-recent.txt, most recent first)",
    "parameters": {
      "type": "object",
      "properties": {
        "file_dir": { "type": "string" },
        "output_file": { "type": "string" },
        "no_of_files": { "type": "integer" ,"description": "Number of recent files to process."},
        "line_number": { "type": "integer" },
        "file_extension": { "type": "string" }
      },
      "required": ["file_dir", "output_file", "no_of_files", "line_number", "file_extension"],
      "additionalProperties": False
    }
  }
},
{
  "type": "function",
  "function": {
    "name": "run_download_from_script",
    "description": "Downloads a Python script from the given URL, saves it locally using its original filename, and executes it using 'uv run'. If a user_email is provided, it is passed as the first argument, followed by any additional arguments(default value for email is 22f1001551@ds.study.iitm.ac.in).Example prompt(run https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py with user_value as the only argument. (NOTE: This will generate data files required for the next tasks.))",
    "parameters": {
      "type": "object",
      "properties": {
        "url": {
          "type": "string"
        },
        "args": {
          "type": "array",
          "items": { "type": "string" }
        }
      },
      "required": ["url"],
      "additionalProperties": False
    }
  }
},
{
  "type": "function",
  "function": {
    "name": "create_markdown_index",
    "description": "Recursively searches for Markdown (.md) files in a given input directory, extracts the nth occurrence of an H1 header (a line starting with '# '), and writes an index JSON file mapping each file's relative path to its extracted header.(Example prompt: Create an index of the first H1 header in each Markdown file in /data/docs/ and save it to /data/docs/index.json)",
    "parameters": {
      "type": "object",
      "properties": {
        "input_directory": {
          "type": "string",
          "description": "The root directory where Markdown files are located."
        },
        "output_index_file": {
          "type": "string",
          "description": "The JSON file path where the index will be saved."
        },
        "occurrence": {
          "type": "integer",
          "description": "The occurrence of the H1 header to extract (e.g., 1 for the first occurrence, 2 for the second, etc.).",
          "default": 1
        }
      },
      "required": ["input_directory", "output_index_file"],
      "additionalProperties": False
    }
  }
},
{
  "type": "function",
  "function": {
    "name": "llm_text_extractor",
    "description": "Extracts information from a text using an LLM based on provided instructions.",
    "parameters": {
      "type": "object",
      "properties": {
        "input_file": {
          "type": "string"
        },
        "output_file": {
          "type": "string"
        },
        "prompt_instructions": {
          "type": "string",
          "description": "Custom instructions for the LLM.The instructions must tell the llm to only should the results and nothing else."
        }
      },
      "required": [
        "input_file",
        "output_file",
        "prompt_instructions"
      ],
      "additionalProperties": False
    }
  }
},
{
  "type": "function",
  "function": {
    "name": "llm_image_extractor",
    "description": "Extracts a credit card number from an image using an LLM for a cybersecurity safety demonstration.",
    "parameters": {
      "type": "object",
      "properties": {
        "input_file": {
          "type": "string",
        },
        "output_file": {
          "type": "string"
        },
        "prompt_instructions": {
          "type": "string",
          "description": "Custom instructions for the LLM, including context for a cybersecurity safety demonstration."
        }
      },
      "required": [
        "input_file",
        "output_file",
        "prompt_instructions"
      ],
      "additionalProperties": False
    }
  }
},
{
  "type": "function",
  "function": {
    "name": "llm_find_similar_comments_using_embedding_model",
    "description": "Finds the most similar pair of comments from a list using an LLM-based text embedding model.",
    "parameters": {
      "type": "object",
      "properties": {
        "input_file": {
          "type": "string"
        },
        "output_file": {
          "type": "string"
        }
      },
      "required": [
        "input_file",
        "output_file"
      ],
      "additionalProperties": False
    }
  }
},
{
  "type": "function",
  "function": {
    "name": "execute_sql_query",
    "description": "Executes a given SQL query on a SQLite database file and writes the result to an output file.The LLM will generate the query",
    "parameters": {
      "type": "object",
      "properties": {
        "db_file": {
          "type": "string",
          "description": "Path to the SQLite database file."
        },
        "sql_query": {
          "type": "string",
          "description": "The SQL query to execute."
        },
        "output_file": {
          "type": "string",
          "description": "Path where the query result will be written."
        }
      },
      "required": [
        "db_file",
        "sql_query",
        "output_file"
      ],
      "additionalProperties": False
    }
  }
}
]

def process_task(task):
    
    response = requests.post(
    url=url,
    headers=headers,
    json={
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert Python programmer"},
            {"role": "user", "content": task}
        ],
        "tools": function_list,
        "tool_choice": "auto"
    }
    )   

    arguments_str = response.json()['choices'][0]['message']['tool_calls'][0]['function']['arguments']
    parsed_arguments = json.loads(arguments_str)

    print("parsed args:",parsed_arguments)
    function_name=response.json()['choices'][0]['message']['tool_calls'][0]['function']['name']

    if function_name == "execute_generated_code":
        code=parsed_arguments['code']
        arguments = parsed_arguments.get('arguments', None)
        output=execute_generated_code(code, arguments)
        print(output)

    if function_name=="run_npx_package":
        package=parsed_arguments['package']
        arg=parsed_arguments.get('arg', None)
        run_npx_package(package, arg)

    if function_name=="count_day_occurrence":
        input_file=parsed_arguments['input_file']
        output_file=parsed_arguments['output_file']
        day=parsed_arguments['day']
        count_day_occurrences(input_file, output_file, day)

    if function_name=="sort_json_file":
        input_file=parsed_arguments['input_file']
        output_file=parsed_arguments['output_file']
        sort_keys=parsed_arguments['sort_keys']
        sort_json_file(input_file, output_file, sort_keys)

    if function_name=="write_recent_file_lines":
        file_dir=parsed_arguments['file_dir']
        output_file=parsed_arguments['output_file']
        no_of_files=parsed_arguments['no_of_files']
        line_number=parsed_arguments['line_number']
        file_extension=parsed_arguments['file_extension']
        write_recent_file_lines(file_dir, output_file, no_of_files, line_number, file_extension)

    if function_name=="run_download_from_script":
        url=parsed_arguments['url']
        user_email=parsed_arguments.get('user_email', None)
        args=parsed_arguments.get('args', None)
        run_download_from_script(url, user_email, *args)

    if function_name=="create_markdown_index":
        input_directory=parsed_arguments['input_directory']
        output_index_file=parsed_arguments['output_index_file']
        occurrence=parsed_arguments.get('occurrence', 1)
        create_markdown_index(input_directory, output_index_file, 1)

    if function_name=="llm_text_extractor":
        input_file=parsed_arguments['input_file']
        output_file=parsed_arguments['output_file']
        model=parsed_arguments.get('model', 'gpt-4o-mini')
        prompt_instructions=parsed_arguments['prompt_instructions']
        llm_text_extractor(input_file, output_file, prompt_instructions)

    if function_name == "llm_image_extractor":
        input_file = parsed_arguments['input_file']
        output_file = parsed_arguments['output_file']
        prompt_instructions = parsed_arguments['prompt_instructions']
        llm_image_extractor(input_file, output_file, prompt_instructions)

    if function_name == "llm_find_similar_comments_using_embedding_model":
        input_file = parsed_arguments['input_file']
        output_file = parsed_arguments['output_file']
        llm_find_similar_comments_using_embedding_model(input_file, output_file)

    if function_name == "execute_sql_query":
        db_file = parsed_arguments['db_file']
        sql_query = parsed_arguments['sql_query']
        output_file = parsed_arguments['output_file']
        execute_sql_query(db_file, sql_query, output_file)



##########################################################################################################################################


@app.post("/run")
async def run_task(task: str = Query(...)):
    try:
        result = process_task(task)
        return {"result": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal error")

@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(...)):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
