# SOICT 2026 - Abstract Submission Package
**Conference**: The 2026 International Symposium on Information and Communication Technology (SOICT 2026)  
**Submission Portal**: [EasyChair SOICT 2026](https://easychair.org/conferences/?conf=soict2026)  
**Abstract Deadline**: **20 September 2026** (Today)  
**Full Paper Deadline**: 20 September 2026 (System open until 25 September 2026)  
**Publication**: Springer CCIS Proceedings (Indexed by Scopus, EI Compendex, SpringerLink)  
**Format**: Springer LNCS/CCIS Template, Single-blind, Max 12 pages (excl. references)

---

## 1. Paper Metadata

### Title
**CliffordTrajNet: Geometric Clifford Representations and Continuous State-Space Dynamics for Rigorous Autonomous Driving and UAV Test Prioritization**

### Recommended Track (Select one during EasyChair submission)
- **Primary**: *Software Engineering, Trusted Digital Platforms, and Smart Services*  
  *(Sub-topics: Software testing, debugging, verification, and validation; Software engineering for AI-based systems; Empirical software engineering)*
- **Alternative**: *Applied AI, Big Data Analytics, and Data-Driven Applications*  
  *(Sub-topics: Applied machine learning and deep learning; Decision support systems; Responsible, explainable, and trustworthy applied AI)*

### Authors & Affiliations (Single-Blind)
1. **Chi-Nguyen Tran** ($^*$Equal contribution)  
   - Affiliation: *Faculty of Information Technology, University of Science, VNU-HCM, Ho Chi Minh City, Vietnam*  
   - Email: `23122044@student.hcmus.edu.vn`
2. **Dao Sy Duy Minh** ($^*$Equal contribution)  
   - Affiliation: *Faculty of Information Technology, University of Science, VNU-HCM, Ho Chi Minh City, Vietnam*  
   - Email: `23122041@student.hcmus.edu.vn`
3. **Huynh Trung Kiet** ($^*$Equal contribution)  
   - Affiliation: *Faculty of Information Technology, University of Science, VNU-HCM, Ho Chi Minh City, Vietnam*  
   - Email: `23122039@student.hcmus.edu.vn`
4. **Phu-Hoa Pham**  
   - Affiliation: *Faculty of Information Technology, University of Science, VNU-HCM, Ho Chi Minh City, Vietnam*  
   - Email: `23122030@student.hcmus.edu.vn`
5. **Nguyen Lam Phu Quy**  
   - Affiliation: *Faculty of Information Technology, University of Science, VNU-HCM, Ho Chi Minh City, Vietnam*  
   - Email: `23122048@student.hcmus.edu.vn`

### Keywords
`Autonomous Driving Testing`, `Aerial Drone Testing`, `Test Case Prioritization`, `Clifford Geometric Algebra`, `Continuous State-Space Models`, `Conformal Risk Control`

---

## 2. Abstract Text (Ready to Copy-Paste into EasyChair)

Simulation-based testing is paramount for assuring the safety and reliability of Autonomous Cyber-Physical Systems (ACPS), spanning autonomous driving systems (ADS) and unmanned aerial vehicles (UAVs). However, executing high-fidelity virtual simulations across vast operational design domains is computationally prohibitive under tight release cycles. Test case prioritization (TCP) orders test executions to detect safety-critical faults as early as possible. Existing heuristic and deep learning prioritizers suffer from severe foundational vulnerabilities: they are sensitive to coordinate frame rotations, incur discretization errors under non-uniform trajectory waypoint sampling, and lack finite-sample statistical safety guarantees.

In this work, we propose **CliffordTrajNet**, a unified geometric deep learning and continuous state-space framework for test prioritization across autonomous vehicles and aerial drones. First, CliffordTrajNet formulates spatial trajectories within Clifford Geometric Algebra $C\ell(p,q)$ ($C\ell(2,0)$ for 2D road driving and $C\ell(3,0)$ for 3D aerial flight), employing multivector representations and rotor sandwich operations to guarantee exact rotational and translational invariance without coordinate distortion or Gimbal Lock. Second, it introduces a continuous-time Selective State Space Model (TrajMamba) driven by arclength ODE integration, eliminating discretization error across variable-resolution waypoint sequences. Third, we establish a finite-sample Conformal Risk Control mechanism providing Probably Approximately Correct (PAC) statistical bounds on fault detection under constrained execution budgets. Extensive empirical evaluation on the real-world SensoDat benchmark (32,580 tests), multiple SDC testbeds, and the official SBFT 2026 UAV Testing Competition benchmark confirms that CliffordTrajNet achieves exact coordinate invariance ($\Delta\mathrm{APFD} = 0.0000$), robust discretization tolerance ($\Delta\mathrm{APFD} \le 0.0012$), and state-of-the-art failure discovery (SDC APFD $0.8063$, AUC $0.9399$; UAV APFD $0.7235$) without heuristic artifacts.
