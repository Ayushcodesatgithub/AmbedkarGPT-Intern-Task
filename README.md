# AmbedkarGPT-Intern-Task

<img width="1078" height="264" alt="image" src="https://github.com/user-attachments/assets/d9072298-3c23-4210-8b5e-b888a6138a83" />
 Key Features:

Persistent Storage: Vector store is saved to disk, so subsequent runs are faster
Interactive Q&A: User-friendly command-line interface
Smart Chunking: Text split into 200-char chunks with 50-char overlap
Custom Prompts: Optimized prompt template for focused answers
Error Handling: Comprehensive error messages and setup validation
Clean Code: Well-commented and follows best practices

System Flow:

Load speech.txt → 2. Split into chunks → 3. Generate embeddings → 4. Store in ChromaDB → 5. User asks question → 6. Retrieve relevant chunks → 7. Feed to Mistral → 8. Get answer!


