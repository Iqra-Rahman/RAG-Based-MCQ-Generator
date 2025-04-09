import json
import math
import random
import time
import re
from typing import Dict, List, Set
import openai
import os
from dotenv import load_dotenv
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

load_dotenv()

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version
)

# Categories and keywords
PREDEFINED_CATEGORIES = {
    "Core Concepts": [
        "Transformer", "Self-Attention", "Multi-Head Attention",
        "Scaled Dot-Product Attention", "Positional Encoding",
        "Feed-Forward Networks", "Encoder-Decoder Architecture"
    ],
    "Training & Optimization": [
        "Adam Optimizer", "Dropout Regularization", "Label Smoothing",
        "Beam Search", "Byte-Pair Encoding (BPE)",
        "Neural Machine Translation (NMT)", "WMT 2014 Dataset", "BLEU Score"
    ],
    "Comparison with Other Models": [
        "Recurrent Neural Networks (RNNs)", "Long Short-Term Memory (LSTMs)",
        "Gated Recurrent Units (GRUs)", "Convolutional Neural Networks (CNNs)",
        "ByteNet", "ConvS2S", "GNMT (Google's Neural Machine Translation System)"
    ],
    "Key Findings & Advantages": [
        "No recurrence, fully attention-based model", "Parallelization in training",
        "Faster training time (12 hours on 8 P100 GPUs)", "Better translation performance",
        "State-of-the-art BLEU scores"
    ],
    "Applications": [
        "Machine Translation", "Constituency Parsing", "Sequence-to-Sequence Learning",
        "Text Processing", "Natural Language Processing (NLP)"
    ]
}

ALL_PREDEFINED_KEYWORDS = [
    keyword for category, keywords in PREDEFINED_CATEGORIES.items()
    for keyword in keywords
]

class MCQItem(BaseModel):
    """Schema for a single MCQ"""
    question: str = Field(description="MCQ question")
    options: Dict[str, str] = Field(description="Answer options A, B, C, D")
    correct_answer: str = Field(description="Correct option (A, B, C, or D)")
    explanation: str = Field(description="Explanation for correct answer")
    source: str = Field(description="Source of the MCQ (PDF or Predefined)")
    difficulty: str = Field(description="Difficulty level (Easy, Moderate, Hard)")

    @field_validator('correct_answer')
    @classmethod
    def validate_correct_answer(cls, v):
        if v not in ['A', 'B', 'C', 'D']:
            raise ValueError("Correct answer must be A, B, C, or D")
        return v

    @field_validator('options')
    @classmethod
    def validate_options(cls, v):
        if not all(key in v for key in ['A', 'B', 'C', 'D']):
            raise ValueError("Options must contain keys A, B, C, and D")
        return v

    @field_validator('difficulty')
    @classmethod
    def validate_difficulty(cls, v):
        if v not in ['Easy', 'Moderate', 'Hard']:
            raise ValueError("Difficulty must be Easy, Moderate, or Hard")
        return v

    @field_validator('source')
    @classmethod
    def validate_source(cls, v):
        if v not in ['PDF', 'Predefined']:
            raise ValueError("Source must be PDF or Predefined")
        return v

class QnAItem(BaseModel):
    """Schema for a single QnA pair"""
    question: str = Field(description="Question")
    answer: str = Field(description="Answer")
    source: str = Field(description="Source of the QnA (Predefined)")
    difficulty: str = Field(description="Difficulty level (Easy, Moderate, Hard)")

    @field_validator('difficulty')
    @classmethod
    def validate_difficulty(cls, v):
        if v not in ['Easy', 'Moderate', 'Hard']:
            raise ValueError("Difficulty must be Easy, Moderate, or Hard")
        return v

    @field_validator('source')
    @classmethod
    def validate_source(cls, v):
        if v != 'Predefined':
            raise ValueError("Source must be Predefined for QnA")
        return v

class PDFMCQGenerator:
    def __init__(self, model="gpt-4o", temperature=0.7):
        """Initialize OpenAI-based MCQ and QnA generator"""
        self.model = model
        self.temperature = temperature
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        self.retry_count = 5
        self.mcqs_per_chunk = 2
        self.successful_chunks = 0
        self.failed_chunks = 0
        self.important_keywords = set()

    def extract_keywords(self, document):
        """Extract important keywords from the document"""
        print("🔑 Extracting important keywords from document...")
        full_text = " ".join([doc.page_content for doc in document])
        full_text = re.sub(r'\[\d+\]|\(\w+ et al\.,? \d{4}\)|\([A-Za-z]+, \d{4}\)', '', full_text)
        prompt = """
        Extract 15-20 most important technical keywords or concepts from this text.
        Focus on subject-specific terminology that represents the core concepts.
        Return ONLY a comma-separated list of these keywords, with no additional text.
        TEXT:
        {text}
        """
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a technical keyword extractor."},
                    {"role": "user", "content": prompt.format(text=full_text[:4000])}
                ],
                temperature=0.3,
                max_tokens=200
            )
            keyword_text = response.choices[0].message.content.strip()
            keywords = [k.strip() for k in keyword_text.split(',')]
            self.important_keywords = set(keywords)
            print(f"✅ Extracted {len(self.important_keywords)} keywords: {', '.join(list(self.important_keywords)[:5])}...")
            return self.important_keywords
        except Exception as e:
            print(f"⚠️ Keyword extraction failed: {str(e)[:50]}...")
            return set()

    def should_exclude_chunk(self, chunk):
        """Check if chunk should be excluded (references, acknowledgements, etc.)"""
        ref_patterns = [
            r'References\s', r'Bibliography\s', r'Acknowledgements\s',
            r'et al\.\s+\(\d{4}\)', r'\[\d+\]\s+[A-Z][a-z]+,',
            r'^\s*\d+\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+et\s+al\.',
        ]
        for pattern in ref_patterns:
            if re.search(pattern, chunk):
                return True
        citation_count = len(re.findall(r'\[\d+\]|\(\w+ et al\.,? \d{4}\)|\([A-Za-z]+, \d{4}\)', chunk))
        text_length = len(chunk)
        if citation_count > 3 and (citation_count * 10 / text_length) > 0.2:
            return True
        return False

    def load_and_split_pdf(self, pdf_path):
        """Load and process a PDF document"""
        print(f"📖 Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        self.extract_keywords(documents)
        all_chunks = []
        for doc in documents:
            if self.should_exclude_chunk(doc.page_content):
                continue
            page_chunks = self.text_splitter.split_text(doc.page_content)
            filtered_chunks = [chunk for chunk in page_chunks if not self.should_exclude_chunk(chunk)]
            all_chunks.extend(filtered_chunks)
        print(f"✅ PDF split into {len(all_chunks)} chunks (after filtering)")
        if len(all_chunks) < 100:
            print("⚠️ Creating additional chunks with different parameters...")
            chunking_strategies = [
                {"chunk_size": 300, "chunk_overlap": 20},
                {"chunk_size": 200, "chunk_overlap": 10},
                {"chunk_size": 500, "chunk_overlap": 100}
            ]
            additional_chunks = []
            for strategy in chunking_strategies:
                splitter = RecursiveCharacterTextSplitter(**strategy)
                for doc in documents:
                    if self.should_exclude_chunk(doc.page_content):
                        continue
                    strategy_chunks = splitter.split_text(doc.page_content)
                    filtered_chunks = [chunk for chunk in strategy_chunks if not self.should_exclude_chunk(chunk)]
                    additional_chunks.extend(filtered_chunks)
            existing_chunks = set(all_chunks)
            for chunk in additional_chunks:
                if chunk not in existing_chunks:
                    all_chunks.append(chunk)
                    existing_chunks.add(chunk)
            print(f"✅ Enhanced PDF split into {len(all_chunks)} chunks (after filtering)")
        return all_chunks

    def clean_json_response(self, content):
        """Clean and parse JSON from API response"""
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.replace(",]", "]").replace(",}", "}")
        if not (content.startswith("[") and content.endswith("]")):
            if content.startswith("{") and content.endswith("}"):
                content = f"[{content}]"
            elif "{" in content and "}" in content:
                parts = []
                depth = 0
                start = -1
                for i, char in enumerate(content):
                    if char == '{':
                        if depth == 0:
                            start = i
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0 and start != -1:
                            parts.append(content[start:i+1])
                            start = -1
                if parts:
                    content = f"[{','.join(parts)}]"
        return content

    def generate_multiple_mcqs(self, chunk, is_predefined=False, category="", difficulty_level=""):
        """Generate multiple MCQs"""
        if is_predefined:
            if category:
                keywords = PREDEFINED_CATEGORIES.get(category, [])
                keyword_text = ", ".join(keywords)
                topic_str = f"the '{category}' category, specifically about: {keyword_text}"
            else:
                selected_keywords = random.sample(ALL_PREDEFINED_KEYWORDS, min(5, len(ALL_PREDEFINED_KEYWORDS)))
                keyword_text = ", ".join(selected_keywords)
                topic_str = f"these specific concepts: {keyword_text}"
            difficulty_str = f"Create {difficulty_level} difficulty questions (avoid highly complex or specialized details)"
            prompt = f"""
            Create exactly {self.mcqs_per_chunk} MCQs about Transformer models and {topic_str}.
            IMPORTANT INSTRUCTIONS:
            1. {difficulty_str}
            2. Questions should be factual and educational
            3. Focus on clear, well-formed questions about fundamental concepts
            4. Ensure the correct answers are accurate and unambiguous
            5. Include explanations that are educational and helpful for learning
            6. DO NOT create questions about specific papers, authors, or citations
            For each MCQ, follow this JSON format EXACTLY:
            {{
                "question": "Clear, concise question about {keyword_text}",
                "options": {{
                    "A": "First option",
                    "B": "Second option",
                    "C": "Third option",
                    "D": "Fourth option"
                }},
                "correct_answer": "A, B, C, or D",
                "explanation": "Brief explanation of why the answer is correct",
                "source": "Predefined",
                "difficulty": "{difficulty_level}"
            }}
            Return a JSON array containing {self.mcqs_per_chunk} MCQ objects.
            """
        else:
            keyword_text = ", ".join(list(self.important_keywords)[:10]) if self.important_keywords else ""
            prompt = f"""
            Create exactly {self.mcqs_per_chunk} difficult MCQs from this text:
            {chunk}
            IMPORTANT INSTRUCTIONS:
            1. DO NOT create questions about authors, citations, publication dates, or references
            2. DO NOT ask about who wrote or published the content
            3. DO NOT create questions that refer to specific citations like [1], [2], etc.
            4. Focus on technical content, concepts, methods, and applications
            5. If possible, focus on these key topics: {keyword_text}
            For each MCQ, follow this JSON format EXACTLY:
            {{
                "question": "Clear, concise question based on the technical content",
                "options": {{
                    "A": "First option",
                    "B": "Second option",
                    "C": "Third option",
                    "D": "Fourth option"
                }},
                "correct_answer": "A, B, C, or D",
                "explanation": "Brief explanation of why the answer is correct",
                "source": "PDF",
                "difficulty": "Hard"
            }}
            Return a JSON array containing {self.mcqs_per_chunk} MCQ objects.
            """
        backoff_time = 2
        for attempt in range(self.retry_count):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an MCQ generator specialized in creating valid JSON and technical questions. Never create questions about paper authors, citations, or references."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                )
                content = response.choices[0].message.content
                if not content:
                    continue
                try:
                    content = self.clean_json_response(content)
                    mcq_list = json.loads(content)
                    if not isinstance(mcq_list, list):
                        mcq_list = [mcq_list]
                    valid_mcqs = []
                    for mcq_data in mcq_list:
                        try:
                            question = mcq_data.get("question", "").lower()
                            author_patterns = [
                                "author", "wrote", "published", "cited", "reference",
                                "et al", "citation", "paper", "article", "researcher",
                                "study by", "according to", "et al."
                            ]
                            if any(pattern in question for pattern in author_patterns):
                                continue
                            if re.search(r'$$ \d+ $$|$$ [A-Za-z]+(,? \w+)* et al\.?,? \d{4} $$', question):
                                continue
                            if "source" not in mcq_data:
                                mcq_data["source"] = "Predefined" if is_predefined else "PDF"
                            if "difficulty" not in mcq_data:
                                mcq_data["difficulty"] = difficulty_level if is_predefined else "Hard"
                            valid_mcqs.append(MCQItem(**mcq_data))
                        except Exception as e:
                            print(f"⚠️ Invalid MCQ format: {str(e)[:50]}...")
                            continue
                    if valid_mcqs:
                        self.successful_chunks += 1
                        return valid_mcqs
                except json.JSONDecodeError as e:
                    print(f"❌ JSON Parse Error: {str(e)[:50]}... (Attempt {attempt+1}/{self.retry_count})")
            except Exception as e:
                print(f"❌ API Error: {str(e)[:50]}... (Attempt {attempt+1}/{self.retry_count})")
            backoff_time = min(60, backoff_time * 1.5)
            sleep_time = backoff_time + random.uniform(0, 2)
            time.sleep(sleep_time)
        self.failed_chunks += 1
        return []

    def generate_multiple_qna(self, is_predefined=True, category="", difficulty_level=""):
        """Generate multiple QnA pairs based on predefined categories"""
        if category:
            keywords = PREDEFINED_CATEGORIES.get(category, [])
            keyword_text = ", ".join(keywords)
            topic_str = f"the '{category}' category, specifically about: {keyword_text}"
        else:
            selected_keywords = random.sample(ALL_PREDEFINED_KEYWORDS, min(5, len(ALL_PREDEFINED_KEYWORDS)))
            keyword_text = ", ".join(selected_keywords)
            topic_str = f"these specific concepts: {keyword_text}"
        difficulty_str = f"Create {difficulty_level} difficulty questions"
        prompt = f"""
        Create exactly {self.mcqs_per_chunk} question and answer pairs about Transformer models and {topic_str}.
        IMPORTANT INSTRUCTIONS:
        1. {difficulty_str}
        2. Questions should be factual and educational
        3. Focus on clear, well-formed questions about fundamental concepts
        4. Ensure the answers are accurate and informative
        5. Include answers that are direct and helpful for learning
        6. DO NOT create questions about specific papers, authors, or citations
        For each QnA, follow this JSON format EXACTLY:
        {{
            "question": "Clear, concise question about {keyword_text}",
            "answer": "Direct and informative answer",
            "source": "Predefined",
            "difficulty": "{difficulty_level}"
        }}
        Return a JSON array containing {self.mcqs_per_chunk} QnA objects.
        """
        backoff_time = 2
        for attempt in range(self.retry_count):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a QnA generator specialized in creating valid JSON and technical questions. Never create questions about paper authors, citations, or references."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                )
                content = response.choices[0].message.content
                if not content:
                    continue
                try:
                    content = self.clean_json_response(content)
                    qna_list = json.loads(content)
                    if not isinstance(qna_list, list):
                        qna_list = [qna_list]
                    valid_qna = []
                    for qna_data in qna_list:
                        try:
                            question = qna_data.get("question", "").lower()
                            author_patterns = [
                                "author", "wrote", "published", "cited", "reference",
                                "et al", "citation", "paper", "article", "researcher",
                                "study by", "according to", "et al."
                            ]
                            if any(pattern in question for pattern in author_patterns):
                                continue
                            if re.search(r'$$ \d+ $$|$$ [A-Za-z]+(,? \w+)* et al\.?,? \d{4} $$', question):
                                continue
                            if "source" not in qna_data:
                                qna_data["source"] = "Predefined"
                            if "difficulty" not in qna_data:
                                qna_data["difficulty"] = difficulty_level
                            valid_qna.append(QnAItem(**qna_data))
                        except Exception as e:
                            print(f"⚠️ Invalid QnA format: {str(e)[:50]}...")
                            continue
                    if valid_qna:
                        self.successful_chunks += 1
                        return valid_qna
                except json.JSONDecodeError as e:
                    print(f"❌ JSON Parse Error: {str(e)[:50]}... (Attempt {attempt+1}/{self.retry_count})")
            except Exception as e:
                print(f"❌ API Error: {str(e)[:50]}... (Attempt {attempt+1}/{self.retry_count})")
            backoff_time = min(60, backoff_time * 1.5)
            sleep_time = backoff_time + random.uniform(0, 2)
            time.sleep(sleep_time)
        self.failed_chunks += 1
        return []

    def generate_predefined_qna(self, target_count, difficulty_distribution=None):
        """Generate QnA pairs from predefined categories"""
        if difficulty_distribution is None:
            difficulty_distribution = {"Easy": 0.4, "Moderate": 0.6}
        print(f"🔄 Generating {target_count} predefined QnA pairs...")
        predefined_qna = []
        categories = list(PREDEFINED_CATEGORIES.keys())
        difficulty_counts = {
            level: math.ceil(target_count * ratio)
            for level, ratio in difficulty_distribution.items()
        }
        total_assigned = sum(difficulty_counts.values())
        if total_assigned > target_count:
            levels = list(difficulty_counts.keys())
            difficulty_counts[levels[-1]] -= (total_assigned - target_count)
        for difficulty, count in difficulty_counts.items():
            remaining = count
            base_per_category = max(1, count // len(categories))
            print(f"📝 Generating {count} {difficulty} difficulty QnA pairs across {len(categories)} categories")
            for category in categories:
                if remaining <= 0:
                    break
                to_generate = min(base_per_category, remaining)
                self.mcqs_per_chunk = min(3, to_generate)
                print(f"  ➡️ Generating {to_generate} QnA pairs for '{category}' category")
                while remaining > 0 and to_generate > 0:
                    batch_size = min(self.mcqs_per_chunk, to_generate)
                    self.mcqs_per_chunk = batch_size
                    new_qna = self.generate_multiple_qna(
                        is_predefined=True,
                        category=category,
                        difficulty_level=difficulty
                    )
                    if new_qna:
                        to_add = min(len(new_qna), to_generate)
                        predefined_qna.extend(new_qna[:to_add])
                        remaining -= to_add
                        to_generate -= to_add
                        time.sleep(1)
                    else:
                        if self.mcqs_per_chunk > 1:
                            self.mcqs_per_chunk -= 1
                        else:
                            break
            while remaining > 0:
                category = random.choice(categories)
                self.mcqs_per_chunk = min(2, remaining)
                new_qna = self.generate_multiple_qna(
                    is_predefined=True,
                    category=category,
                    difficulty_level=difficulty
                )
                if new_qna:
                    to_add = min(len(new_qna), remaining)
                    predefined_qna.extend(new_qna[:to_add])
                    remaining -= to_add
                    time.sleep(1)
                else:
                    continue
        print(f"✅ Generated {len(predefined_qna)}/{target_count} predefined QnA pairs")
        return predefined_qna

    def generate_mcqs(self, pdf_path, target_count=150, predefined_percentage=0.3):
        """Generate MCQs from PDF and QnA pairs from predefined categories"""
        predefined_count = int(target_count * predefined_percentage)
        pdf_count = target_count - predefined_count
        print(f"📋 Target counts: {pdf_count} MCQs from PDF, {predefined_count} QnA from predefined keywords (30%)")

        predefined_qna = self.generate_predefined_qna(
            predefined_count,
            difficulty_distribution={"Easy": 0.4, "Moderate": 0.6}
        )

        original_chunks = self.load_and_split_pdf(pdf_path)
        chunks = original_chunks.copy()
        pdf_mcqs = []
        random.shuffle(chunks)

        with tqdm(total=pdf_count, desc="🔄 Generating PDF MCQs") as pbar:
            chunk_idx = 0
            while len(pdf_mcqs) < pdf_count and chunk_idx < len(chunks):
                chunk = chunks[chunk_idx]
                new_mcqs = self.generate_multiple_mcqs(chunk)
                for mcq in new_mcqs:
                    if len(pdf_mcqs) < pdf_count:
                        pdf_mcqs.append(mcq)
                        pbar.update(1)
                    else:
                        break
                if chunk_idx > 0 and chunk_idx % 10 == 0:
                    success_rate = self.successful_chunks / (self.successful_chunks + self.failed_chunks) if (self.successful_chunks + self.failed_chunks) > 0 else 0
                    if success_rate < 0.5:
                        self.mcqs_per_chunk = max(1, self.mcqs_per_chunk - 1)
                        print(f"⚠️ Adjusting to {self.mcqs_per_chunk} MCQs per chunk due to low success rate ({success_rate:.2f})")
                    elif success_rate > 0.8 and self.mcqs_per_chunk < 3:
                        self.mcqs_per_chunk += 1
                        print(f"✓ Increasing to {self.mcqs_per_chunk} MCQs per chunk due to high success rate ({success_rate:.2f})")
                if chunk_idx % 5 == 0:
                    print(f"Progress: {len(pdf_mcqs)}/{pdf_count} MCQs ({chunk_idx+1}/{len(chunks)} chunks processed)")
                chunk_idx += 1
                time.sleep(1)

            if len(pdf_mcqs) < pdf_count:
                print(f"⚠️ Only generated {len(pdf_mcqs)}/{pdf_count} MCQs in first pass. Starting second pass...")
                second_pass_chunks = original_chunks.copy()
                random.shuffle(second_pass_chunks)
                self.mcqs_per_chunk = 1
                self.temperature = 0.9
                chunk_idx = 0
                while len(pdf_mcqs) < pdf_count and chunk_idx < len(second_pass_chunks):
                    chunk = second_pass_chunks[chunk_idx]
                    new_mcqs = self.generate_multiple_mcqs(chunk)
                    for mcq in new_mcqs:
                        if len(pdf_mcqs) < pdf_count:
                            pdf_mcqs.append(mcq)
                            pbar.update(1)
                        else:
                            break
                    chunk_idx += 1
                    time.sleep(1.5)

        print(f"✅ Generated {len(pdf_mcqs)} PDF MCQs and {len(predefined_qna)} predefined QnA pairs")
        return pdf_mcqs, predefined_qna

    def save_to_csv(self, items, output_file):
        """Save items to a CSV file"""
        if not items:
            print("❌ No items to save")
            return
        df = pd.DataFrame([item.model_dump() for item in items])
        df.to_csv(output_file, index=False)
        print(f"✅ {len(items)} items saved to {output_file}")

    def save_to_json(self, items, output_file):
        """Save items in JSON format"""
        if not items:
            print("❌ No items to save")
            return
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([item.model_dump() for item in items], f, indent=4)
        print(f"✅ {len(items)} items saved to {output_file}")

if __name__ == "__main__":
    pdf_path = "/content/attention.pdf"
    generator = PDFMCQGenerator()
    pdf_mcqs, predefined_qna = generator.generate_mcqs(pdf_path, target_count=150, predefined_percentage=0.3)
    generator.save_to_csv(pdf_mcqs, "pdf_mcqs.csv")
    generator.save_to_json(pdf_mcqs, "pdf_mcqs.json")
    generator.save_to_csv(predefined_qna, "predefined_qna.csv")
    generator.save_to_json(predefined_qna, "predefined_qna.json")