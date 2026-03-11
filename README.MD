# Application of Machine Learning Methods for Audio Transcription, Speaker Diarization, and Noise Reduction

## General Information

**Student:** Дмитрин Платон Александрович
**Academic Supervisor:** Ильвовский Дмитрий Алексеевич  

## Project Description

For this master's thesis a web service for audio processing using machine learning methods is planned to be developed.  
The service is expected to solve the following tasks:

- audio transcription;
- speaker diarization;
- audio noise reduction.

This project includes 
- studying of existing methods and models;
- selecting suitable quality evaluation metrics;
- datasets preparation for testing and development;
- developing backend architecture for our service;
- ... 

## Goal of the Thesis

To develop a web service that applies machine learning methods for automatic audio transcription, speaker diarization, and noise reduction.

## Main Objectives

1. Conduct a literature review of existing approaches and open-source solutions in the following areas:
   - automatic speech recognition;
   - speaker diarization;
   - audio noise reduction;
2. Select and describe the data and application scenarios to be used;
3. Perform collection, preparation, and analysis of audio data;
4. Select quality evaluation metrics for each subtask;
5. Implement a baseline solution for audio processing;
6. Improve quality of the ML part of the work;
7. Develop backend solution for our service;
8. Develop a web service that combines the selected models into a unified pipeline;
9. Perform testing, compare solutions, and refine the system.

## Work Plan

### Checkpoint 1. Project Initialization -- Done.
- create the repository;
- prepare the README file;
- formulate a preliminary work plan;
- define the problem statement and project scope.

### Checkpoint 2. Literature Review and Research Foundation -- 12 march 2026
- review research papers, open-source solutions, and existing services;
- analyze modern approaches to transcription, diarization, and noise reduction;
- make a preliminary selection of models, libraries, and tools;
- identify metrics and comparison criteria.
- select open datasets and test audio recordings;

### Checkpoint 3. Data Preparation and Analysis -- 13 march 2026
- analyze formats and characteristics of audio data;
- perform preprocessing, unification, and structuring of the data;
- conduct EDA;
- prepare datasets for testing and baseline experiments.

### Checkpoint 4. Baseline Development and Metric Selection -- 14 march 2026
- choose quality metrics for transcription, diarization, and noise reduction;
- implement a baseline audio processing pipeline;
- obtain initial results on test data;
- analyze the limitations of the baseline approach.

### Checkpoint 5. Improved Solution Development -- 15 march
- select and integrate more advanced models or configurations;
- compare them with the baseline;
- optimize system quality and/or processing speed.

### Checkpoint 6. Web Service Development -- TBA
- design the service architecture;
- implement the backend;
- implement the user interface;
- integrate the models into a unified audio processing service.

### Checkpoint 7. Testing and Refinement
- perform functional testing of the service;
- test the service on different types of audio recordings;
- analyze the quality of the results;
- refine the interface and processing logic.

### More Checkpoints will be added later...

### Checkpoint XXX. Thesis Writing and Final Materials Preparation
- document the research results;
- describe the architecture, experiments, and conclusions;
- prepare demonstration materials.

## Preliminary Data Description

The project is expected to use audio recordings containing:
- speech from one or multiple speakers;
- recordings with different levels of background noise;
- audio of varying duration and quality;
- scenarios close to practical web service use cases.

## Technology Stack

- Python
- Golang
- REST API
- PostreSQL (may change later)
- Docker
- Redis (?)
- ASR models for transcription (edit later)
- speaker diarization tools (edit later)
- noise reduction models or libraries (edit later)
- simple HTTP + JS (?)
- git for version control

## Current Status

The project is currently at the initialization and planning stage.  
Current stage: **Checkpoint 1 — Project Initialization**.