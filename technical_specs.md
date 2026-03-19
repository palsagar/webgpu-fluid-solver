# Architecture Specification: Client-Side GPU Compute System

## 1. System Overview

This document outlines the technical stack and execution model for a distributed web application designed to perform computationally intensive, highly parallelized mathematics. The architecture heavily relies on edge computing, offloading the primary mathematical execution loop to the client's local Graphics Processing Unit (GPU) via modern web APIs, while utilizing a lightweight Python backend for orchestration, state management, and initial data provisioning.

## 2. Frontend Architecture (Client-Side)

The frontend is responsible for both the mathematical execution loop and the visual representation of the data.

### 2.1. Core Technologies

* **Graphics & Compute Wrapper:** Three.js (WebGPU Renderer)
* **Compute API:** WebGPU
* **Shader Language:** WGSL (WebGPU Shading Language) or TSL (Three.js Shading Language)

### 2.2. GPU Compute Execution Model

The system leverages WebGPU Compute Shaders to perform highly parallelized and vectorized mathematical operations. Unlike the standard rendering pipeline (Vertex -> Fragment), the compute pipeline is entirely dedicated to arbitrary data processing.

* **Massively Parallel Execution:** The mathematical model is broken down into discrete operations that can be executed independently across thousands or millions of data points simultaneously. The GPU dispatches compute "workgroups," assigning individual threads to process localized chunks of data in parallel.
* **Vectorized Mathematics:** Shaders are natively optimized for vector and matrix operations (e.g., `vec3`, `mat4`). Complex linear algebra calculations are executed in a single clock cycle at the hardware level.
* **Data Locality & Storage Buffers:** To prevent the severe latency of transferring data back and forth between the CPU (JavaScript) and the GPU, state data is maintained persistently in GPU VRAM using **Storage Buffers**.
* **The Execution Loop:** 1.  The JavaScript thread dispatches a compute pass.
    2.  The Compute Shader reads the current state from the Storage Buffer.
    3.  The highly parallelized math executes, calculating the next step.
    4.  The results are written directly back to the Storage Buffer.
    5.  (Optional) The rendering pipeline reads from this same buffer to draw the results, entirely bypassing the CPU.

## 3. Backend Architecture (Server-Side)

The backend acts as the authoritative control plane. Because real-time network streaming of millions of continuously calculating data points is impossible due to bandwidth constraints, the backend is strictly decoupled from the real-time execution loop.

### 3.1. Core Technologies

* **Framework:** FastAPI (Python)
* **Server:** Uvicorn (ASGI)
* **Data Processing:** NumPy / SciPy (for one-time, pre-computation tasks)

### 3.2. Backend Responsibilities

* **Initial Condition Generation:** Generating complex starting datasets, matrices, or environmental parameters that are too slow to generate sequentially on the client's CPU before uploading to the GPU.
* **State Persistence & Snapshots:** Saving the serialized output of a simulation or calculation at specific intervals to a database for later retrieval.
* **Control Plane API:** Exposing endpoints to alter the overarching parameters of the math running on the client (e.g., updating a global multiplier or friction coefficient).
* **Authentication & Session Management:** Securing access to the compute application.

## 4. Network & Communication Layer

* **RESTful API:** Used for fetching initial heavy datasets (via standard GET requests) and saving end-states (POST/PUT). Payload formats are typically binary (e.g., `Float32Array` buffers) rather than JSON to maximize deserialization speed when loading directly into GPU memory.
* **WebSockets (Optional):** Implemented if the system requires live parameter tuning from a centralized server or multiple collaborative clients. WebSockets transmit lightweight "control signals" (Uniforms) rather than raw positional data.

## 5. Performance Characteristics & Trade-offs

* **Pros:** Achieves supercomputer-like parallel processing speeds by utilizing the user's native hardware. Server costs remain extremely low as the backend performs almost no real-time math.
* **Cons:** Application performance is strictly bound by the quality of the client's GPU. Mobile devices or older integrated graphics cards will experience varying capabilities compared to discrete desktop GPUs.

***

Would you like me to outline the specific data structures (such as Storage Buffers and Uniforms) required to pass the initial parameters from the JavaScript CPU thread into the WebGPU compute shaders?
