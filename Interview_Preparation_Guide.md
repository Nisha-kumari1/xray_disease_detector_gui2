# 🚀 AuraRad: Interview Preparation Guide

This document contains everything you need to know about your project to confidently discuss it in a technical interview. Review these concepts, architectural decisions, and potential interview questions.

---

## 1. Project Overview & Elevator Pitch

**The Problem:** While AI can accurately predict diseases from medical images, it acts as a "black box." Doctors and patients cannot blindly trust an AI without understanding *why* it made a specific diagnosis.
**The Solution (Your Project):** You built **AuraRad**, an end-to-end, clinical-grade Medical Imaging System that prioritizes **Explainable AI (XAI)** and accessibility. It not only predicts Pneumonia with high accuracy but visually proves its reasoning using Grad-CAM heatmaps, and provides a private, local LLM chatbot to explain the results in plain English.

---

## 2. Architecture & Technology Stack

You upgraded a legacy monolithic script into a modern, decoupled, production-ready microservice architecture.

### **Backend (The Engine)**
*   **Framework:** `FastAPI` (Python). Chosen for its extreme speed, asynchronous capabilities, and automatic API documentation (Swagger UI).
*   **Database:** `SQLAlchemy` ORM with SQLite (easily swappable to PostgreSQL). Stores patient scans, predictions, and user data.
*   **Security:** JWT (JSON Web Tokens) for secure, stateless authentication.

### **AI Core (The Brain)**
*   **Disease Detection:** `TensorFlow/Keras` utilizing a deep learning model (e.g., DenseNet121 architecture) trained on Chest X-Rays to classify images as 'Normal' or 'Pneumonia'.
*   **Explainable AI:** `tf.GradientTape` is used to implement **Grad-CAM** (Gradient-weighted Class Activation Mapping). 
*   **Clinical Assistant:** `GPT4All` (specifically the `falcon.Q4_0.gguf` model). A local Large Language Model (LLM) that runs entirely on the CPU/GPU without sending sensitive patient data to the cloud (HIPAA compliance).

### **Frontend (The Interface)**
*   **Framework:** `Next.js 14` (React) chosen for server-side rendering, SEO, and fast page loads.
*   **Styling:** `Tailwind CSS` for utility-first styling, enabling the rapid development of the premium "Clinical Light Theme."
*   **Animations:** `Framer Motion` for smooth, modern UI transitions.

---

## 3. Deep Dive into Key Technical Concepts (Must Know)

If an interviewer asks you to explain the hard technical parts, use these explanations:

### **A. How does Grad-CAM work?**
*   **Concept:** It stands for Gradient-weighted Class Activation Mapping. It tells us which pixels in the image were most important for the model's final decision.
*   **Mechanism:** It looks at the **last convolutional layer** of the neural network. By calculating the gradient (the mathematical derivative) of the predicted class score with respect to the feature maps of that last layer, we can see which spatial features the network was paying attention to. We multiply these gradients by the feature maps, apply a ReLU (to only keep positive influences), and project it as a heatmap over the original X-ray.

### **B. Why use a Local LLM (GPT4All) instead of OpenAI API?**
*   **Data Privacy & Compliance:** In healthcare, sending patient X-rays or queries to a third-party cloud server (like OpenAI) violates HIPAA and patient confidentiality laws. By running GPT4All locally on the machine, the data never leaves the hospital's network.

### **C. Why did you switch from Flask to FastAPI?**
*   **Performance:** FastAPI is built on ASGI (Asynchronous Server Gateway Interface), making it significantly faster than Flask (which is synchronous by default). 
*   **Type Hinting & Validation:** FastAPI uses `Pydantic` for strict data validation. If a frontend sends the wrong data type, FastAPI rejects it automatically, preventing backend crashes.

---

## 4. Common Interview Questions & Answers

**Q: "What CNN architecture did you use — built from scratch or transfer learning?"**
> **A:** "I built a **Custom CNN architecture from scratch**, utilizing **Depthwise Separable Convolutions** and Batch Normalization, rather than relying on Transfer Learning."

**Q: "Why didn't you use Transfer Learning (like ResNet or VGG)?"**
> **A:** I chose a custom architecture for three main reasons:
> 1. **Domain Mismatch:** Models like ResNet are pre-trained on ImageNet (everyday objects in color). Medical X-Rays are grayscale and have fundamentally different, very faint textures (like ground-glass opacities). The feature extractors learned from dogs and cars aren't always optimal for lungs.
> 2. **Preventing Overfitting:** Massive pre-trained models have tens of millions of parameters. Medical datasets are often small, meaning those huge models are highly prone to overfitting. A custom, shallower CNN has far fewer parameters, acting as a natural regularizer.
> 3. **Computational Efficiency:** By using **Depthwise Separable Convolutions** instead of standard convolutions, I drastically reduced the computational cost. This allows the model to perform fast inference on standard hospital CPUs without requiring expensive cloud GPUs."

**Q: "Walk me through what happens when a user uploads an image."**
> **A:** "When the user drags and drops an X-ray, the Next.js frontend sends a multipart POST request to the FastAPI backend. The backend saves the image and passes it to the TensorFlow model. The model calculates the probability of Pneumonia. Simultaneously, the Grad-CAM algorithm calculates the activation heatmap and saves it using OpenCV. Finally, the backend returns the diagnosis, confidence score, and the URLs to both the original image and the heatmap back to the frontend to be displayed side-by-side."

**Q: "How did you handle the class imbalance in your medical dataset?"**
> *(Note: You should adapt this based on how the original model was trained)*
> **A:** "Medical datasets usually have way more 'Normal' cases than 'Disease' cases. To fix this, techniques like assigning Class Weights during training (penalizing the model more for missing a Pneumonia case) and Data Augmentation (rotating, zooming, and flipping the Pneumonia images) were used so the neural network didn't become biased towards just guessing 'Normal'."

**Q: "What was the hardest challenge you faced while building this?"**
> **A:** "Integrating the Local LLM (GPT4All) alongside TensorFlow. Both are heavily resource-intensive. Managing memory allocation and resolving dependency conflicts between the specific versions of Python bindings for the `.gguf` files while keeping the FastAPI server responsive required careful debugging and asynchronous handling."

**Q: "How would you scale this system for a large hospital?"**
> **A:** "I would containerize the backend and frontend using Docker (which I have already prepped). I would deploy the FastAPI backend on a Kubernetes cluster with GPU nodes for fast model inference. I would swap SQLite for a managed PostgreSQL database, and use an AWS S3 bucket to store the actual X-Ray images instead of local storage."

---

## 5. Things to Highlight on your Resume
*   **"Engineered an end-to-end Explainable AI (XAI) medical imaging pipeline using TensorFlow, FastAPI, and Next.js, replacing legacy monolithic architecture."**
*   **"Implemented Grad-CAM algorithms to dynamically generate visual heatmaps, increasing diagnostic transparency for medical professionals."**
*   **"Integrated local, privacy-preserving Large Language Models (GPT4All) to act as an offline clinical assistant, ensuring HIPAA-compliant data handling."**
