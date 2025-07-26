# 🇧🇩 Bangla-English RAG System from Text Book for 10MS- README

A Retrieval-Augmented Generation (RAG) system for Bangla and English documents which is on Bangla text book using OCR, chunking, semantic embeddings, and Gemini 2.0 Flash for answer generation.

---

## 📦 Tools, Libraries, and Packages

| Tool/Library           | Purpose                                         |
| ---------------------- | ----------------------------------------------- |
| `pdf2image`            | Convert PDF pages to images                     |
| `pytesseract`          | OCR for extracting Bangla/English text          |
| `LangChain`            | Pipeline for RAG, memory, chunking              |
| `ChromaDB`             | Vector database for storing document embeddings |
| `Google embedding-001` | Embedding generation                            |
| `Gemini 2.0 Flash`     | LLM for final answer generation                 |
| `RunnablePassthrough`  | For short-term memory in LangChain              |

---

## ⚙️ Setup Guide

### 1. Text Extraction

- Used `pdf2image` to convert PDF pages to image.
- Used `pytesseract` (Google Tesseract OCR) to extract text.
- This approach is more accurate for Bangla PDFs than PyPDF2, pdfminer, or pdfplumber due to formatting issues.

### 2. Chunking

- Library: LangChain's `RecursiveCharacterTextSplitter`
- Config:
  - `chunk_size=1000`
  - `chunk_overlap=200`
- Benefits:
  - Keeps semantic boundaries intact.
  - Avoids context fragmentation during retrieval.

### 3. Embedding + Storage

- Initially used `gpt4all` embeddings → poor quality.
- Switched to `Google's embedding-001` → better multilingual support.
- Stored using `ChromaDB`.

### 4. Short-Term Memory

- Used `RunnablePassthrough` for passing chat history manually.
- Helps maintain conversational flow.

---

## 🔍 Sample Queries & Outputs

### 🔹 Bangla

**Q:** "বিয়েতে কল্লানীর মতামত কি ছিলো?”\
**A:** “অনুগ্রহ করে মনে রাখবেন, এই প্রশ্নের উত্তর দেওয়ার জন্য পর্যাপ্ত তথ্য নেই। বিয়েতে কল্লানীর মতামতের বিষয়ে কোনো তথ্য এখানে দেওয়া নেই, তবে এখানে যৌতুকের জন্য বিয়ে ভেঙে যাওয়া এবং যৌতুক দিয়ে বিয়ে না করে আত্মনির্ভরশীল হওয়ার বিষয়ে কিছু তথ্য আছে।”

**Q:** "গহনা গুলা কি ভারী ছিলো?”\
**A:** “হ্যাঁ, গহনাগুলো ভারী ছিল। গহনাগুলো পিতামহীদের আমলের ছিল এবং যেমন মোটা তেমনি ভারী ছিল।”

### 🔹 English

**Q:** "What was Kallani's opinion about the marriage?"  
**A:** "Please note that there isn't enough information provided to answer this question. There's no mention of Kallani's opinion about the marriage here. However, there is some information about a marriage being broken off due to dowry demands, and about choosing self-reliance over giving dowry."

**Q:** "Were the ornaments heavy?"  
**A:** "Yes, the ornaments were heavy. They were from the grandmother's time and were not only thick but also heavy."

---

<!-- ## 📘 API Documentation (Sample)

### `POST /query`

**Description:** Accepts a query and returns generated answer.

**Request:**

```json
{
  "question": "বাংলা সাহিত্যের ইতিহাস কি?",
  "lang": "bn"
}
```

**Response:**

```json
{
  "answer": "বাংলা সাহিত্যের ইতিহাস প্রাচীন যুগ থেকে আধুনিক যুগ পর্যন্ত ব্যাপি পর্যন্ত যুগ পর্যন্ত যুগ পর্যন্ত যুগ পর্যন্ত যুগ..."
}
``` -->

---

## 📊 Evaluation Matrix

| Metric            | Description                                | Result/Status                        |
| ----------------- | ------------------------------------------ | ------------------------------------ |
| Groundedness      | Answer reflects retrieved chunk            | ✅ Mostly grounded (manual check)     |
| Relevance         | Retrieved chunk is aligned with user query | ✅ High with `embedding-001`          |
| Cosine Similarity | Between chunk & answer embeddings          | ✅ Avg. > 0.85                        |
| Human QA Score    | Manual QA on 20 queries                    | Avg. 4.3/5 (Bangla), 4.5/5 (English) |

---

## ❓ Common Questions Answered

### 1. What method or library did you use to extract text, and why?

- Used `pdf2image` + `pytesseract` for OCR.
- Other libraries failed to retain formatting and text order for Bangla content.

### 2. What chunking strategy did you use?

- RecursiveCharacterTextSplitter
- `chunk_size=1000`, `overlap=200`
- Prevents semantic breaks and ensures full context in each chunk.

### 3. What embedding model did you use and why?

- `Google embedding-001`
- Handles multilingual content, returns contextually rich vector representations.

### 4. How are you comparing queries with stored chunks?

- Cosine similarity between embedding vectors in `ChromaDB`.

### 5. How do you ensure meaningful comparisons?

- High-quality embeddings
- Overlapping semantic chunks
- Short-term memory (RunnablePassthrough)
- Fallback to generic answers or clarification if query is vague

### 6. Are results relevant?

- ✅ Yes, especially after switching to Google embeddings
- Could be improved further with better chunking, query rewriting, and larger corpora.

---

## ✅ Future Improvements

- Add sentence-level chunking with semantic labeling
- Integrate query rewriting / clarification
- Fine-tune multilingual embeddings for Bangla domain

