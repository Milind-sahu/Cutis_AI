---
title: Cutis AI
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Cutis AI - Skin Disease Detection

A Deep Learning based system for detecting skin diseases.

## Overview
This project uses a Convolutional Neural Network (CNN) to analyze images of skin lesions and provide predictions for various skin conditions.

## Project Structure
- `backend/`: FastAPI server for handling predictions and serving the model.
- `frontend/`: Web interface for users to upload images and view results.

## Setup
1. **Backend**:
   - Navigate to `backend/`
   - Create a virtual environment: `python -m venv venv`
   - Activate it: `venv\Scripts\activate` (Windows)
   - Install dependencies: `pip install -r requirements.txt`
   - Run the server: `python run.py`

2. **Frontend**:
   - Open the frontend files in a web browser or serve them using a local server.

## Model
The model is stored in `backend/trained_model/SKIN_MODEL_BEST.keras`. This repository uses **Git LFS** to manage this large file.

## License
MIT
