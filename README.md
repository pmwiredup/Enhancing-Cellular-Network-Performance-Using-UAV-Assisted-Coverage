# Enhancing-Cellular-Network-Performance-Using-UAV-Assisted-Coverage

## Overview

This repository contains the MATLAB simulation code, documentation, and project report for **"Enhancing Cellular Network Performance Using UAV-Assisted Coverage"**—a research project conducted as a part of my Research Internship at IIT Kanpur. The work explores advanced techniques to improve cellular network coverage and reliability by leveraging drone-mounted unmanned aerial vehicles (UAVs) as aerial base stations in a 19-cell hexagonal network. The project features dynamic user clustering, UAV optimization, antenna diversity, and a comprehensive performance evaluation framework.

---

## Table of Contents

- [Overview](#overview)
- [Project Motivation](#project-motivation)
- [Features](#features)
- [System Model](#system-model)
- [Simulation Workflow](#simulation-workflow)
- [Key Results](#key-results)
- [Acknowledgments](#acknowledgments)

---

## Project Motivation

Modern wireless networks face persistent challenges in delivering consistent coverage, especially in high-density urban areas, clustered environments, and emergency situations. Traditional terrestrial base stations often leave coverage holes due to uneven user distribution or urban obstacles. UAVs offer a flexible, rapidly deployable solution to extend and enhance network connectivity where and when it is most needed.

---

## Features

- **19-Cell Hexagonal Network Simulation**: Models realistic cellular layouts, frequency reuse, and inter-cell interference.
- **User Clustering**: Simulates hotspot, commercial, residential, and sparse user distributions using probabilistic models.
- **UAV Optimization**: Dynamically allocates UAVs to maximize coverage and SINR, using genetic algorithms for spatial and altitude optimization.
- **Antenna Diversity (Alamouti Coding)**: Supports link-level diversity techniques for robust communication.
- **Performance Metrics**: Calculates SINR, throughput, and user assignment (BS vs. UAV) for all users.
- **Cluster-Based Analysis**: Breaks down network performance by user cluster type.

---

## System Model

| Parameter                | Value / Description                    |
|--------------------------|----------------------------------------|
| Number of Cells          | 19 (hexagonal layout)                  |
| Cell Radius              | 1000 meters                            |
| Frequency Reuse Factor   | 7                                      |
| Bandwidth                | 10 MHz                                 |
| Path Loss Exponent       | 4 (urban)                              |
| UAV Altitude Range       | 10–120 meters                          |
| UAV Transmit Power       | Up to 23 dBm                           |
| Carrier Frequency        | 2 GHz                                  |
| Average User Density     | 50 users/km² (adjusted by clustering)  |
| SINR Threshold           | 10 dB (for UAV assignment)             |

---

## Simulation Workflow

1. **Cell and Base Station Placement**: Generates a 19-cell hexagonal grid and places base stations.
2. **User Distribution**: Assigns users to cells using Poisson processes and cluster-type probabilities.
3. **UAV Deployment**: Optimizes UAV positions (x, y, altitude) in each cell using a genetic algorithm, weighted by user cluster importance.
4. **Performance Evaluation**: Calculates SINR and throughput for each user from both BS and UAV; assigns users to the best-serving node.
5. **Analysis and Visualization**: Provides cluster-wise performance breakdowns and visual plots of network structure and results.

---

## Key Results

- **Coverage Improvement**: UAVs serve users in coverage holes or dense clusters, improving overall network reach.
- **SINR and Throughput Gains**: Users reassigned to UAVs experience significant improvements in SINR and data rates, especially in hotspot and commercial clusters.
- **Dynamic Adaptation**: The network adapts to real-time changes in user distribution, maintaining high-quality service.

## Acknowledgments

I extend my sincere gratitude to my mentor and the faculty at IIT Kanpur for their invaluable guidance and support throughout this project. Their expertise and encouragement were instrumental in shaping the direction and outcomes of my research. I am also thankful to my peers for their constructive feedback and stimulating discussions, which greatly enriched my learning experience.
