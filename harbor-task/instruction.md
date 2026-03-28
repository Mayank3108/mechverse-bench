# MechVerse: Mechanical Assembly Spatial Reasoning

You are given an image of a 3D mechanical assembly rendered in a CAD environment. Each part in the assembly is colored distinctly (e.g., Red, Blue, Yellow, Orange, Green, Pink, Light Blue). A coordinate axes indicator (X, Y, Z) is visible in the image.

## Your Task

Answer the multiple-choice question about the mechanical assembly shown in the image.

The question will test one of the following reasoning abilities:
- **Path / Disassembly Reasoning** — Can a part be removed? In which direction? What blocks it?
- **Sequence / Precedence Reasoning** — Which part must be removed first? How many parts block a given part?
- **Motion / Kinematic Reasoning** — How does a part move? Along which axis? Does motion couple between parts?
- **Joint / Constraint Reasoning** — What type of joint connects two parts? Along which axis? Where is it located?

## Instructions

1. Study the image carefully, paying attention to:
   - The coordinate axes (X, Y, Z directions)
   - The spatial relationships between colored parts
   - How parts constrain or block each other

2. Read the question and all four answer choices

3. Select the single best answer

4. Write your answer as a single letter: A, B, C, or D

## Answer Format

Your response must contain exactly one character: A, B, C, or D.

Example:
```
B
```

Do not include any explanation, punctuation, or extra text in the answer file.

## Question

The question and answer choices are provided in `dataset\annotations.json`.
The image is provided at `dataset\Images`.
