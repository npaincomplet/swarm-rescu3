# Swarm-Rescue — Design & Scientific Context

## 1. Project Overview

I am participating in the **Swarm-Rescue** competition, where the goal is to design the autonomous behavior of a **swarm of 10 drones** operating in a simulated disaster environment.

The mission is to:

* Explore an **unknown 2D environment**
* Detect **wounded people**
* Transport them to a **rescue center**
* Maximize a **multi-criteria score**

All drones are fully autonomous, identical in code, and evaluated on **previously unseen maps**.

This chat is primarily used for **algorithmic design, scientific reasoning, and strategic decision-making**, not low-level implementation.

---

## 2. Evaluation & Objectives

### Mission termination

A mission ends when either:

* A maximum number of simulation timesteps is reached, or
* A maximum real execution time (walltime) is exceeded

Efficient algorithms reach the timestep limit first; slow ones hit walltime.

### Scoring criteria

The final score combines:

* **Rescue performance**: percentage of wounded people rescued
* **Exploration**: percentage of map explored
* **Drone survivability**: remaining health of drones (especially those returning to the return area)
* **Efficiency**: remaining time if all objectives are completed early

Trade-offs between exploration, rescue, risk, and speed are central.

---

## 3. Environment & Uncertainty

### Partial observability

* Maps are unknown at evaluation time
* Drones perceive only through noisy onboard sensors
* No access to ground-truth state during normal operation

### Dynamic and adversarial conditions

* Possible **loss of GPS**
* Possible **loss of inter-drone communication**
* Presence of **lethal zones** that instantly destroy drones
* Moving (dynamic) wounded persons in some scenarios

Robustness to degraded sensing and communication is essential.

---

## 4. Drone Capabilities (Abstract View)

### Sensors (noisy)

* **Lidar**: geometric obstacle detection (360°)
* **Semantic sensor**: detects wounded persons, rescue center, and other drones
* **GPS & compass**: absolute position and orientation (may be disabled)
* **Odometer**: relative motion estimation (always available, but drifts)

### Actuation

* Continuous motion control (forward, lateral, rotation)
* Binary grasper for carrying wounded persons

### Communication

* Local broadcast to nearby drones within a fixed range
* Messages exchanged at every timestep
* Communication may be disabled in some zones

---

## 5. Programming Model (Conceptual)

* A **single drone policy** is instantiated on all 10 drones
* Each drone runs a perception–decision–action loop
* Decisions must be based **only on local sensor data and received messages**
* Coordination must emerge from decentralized logic

There is **no central controller**.

---

## 6. Core Research & Design Challenges

This project emphasizes:

* **Decentralized multi-agent coordination**
* **Exploration vs. exploitation trade-offs**
* **Robustness to partial failure** (sensor loss, drone loss, comm loss)
* **Implicit role allocation** (explorers, carriers, relays, etc.)
* **Scalable information sharing** under communication constraints
* **Safe navigation** in cluttered and unknown environments

The key difficulty is not implementation complexity, but **designing behaviors that generalize to unseen maps**.

---

## 7. What I Expect from ChatGPT in This Project

Use ChatGPT primarily for:

* High-level **algorithm design**
* **Swarm coordination strategies**
* **Scientific framing** (e.g., inspiration from robotics, control, MARL, swarm intelligence)
* **Failure mode analysis**
* **Design trade-off discussions**
* Conceptual validation of ideas

Avoid focusing on:

* Syntax-level Python questions
* Boilerplate implementation
* GUI or visualization details

---

## 8. Guiding Principle

> **Design for robustness and generalization, not for a specific map or scenario.**

Solutions should remain effective under uncertainty, noise, partial observability, and drone loss.

---

*End of context document*
