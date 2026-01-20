# Awesome ML Serialization Security

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of resources, tools, research papers, and best practices focused on security aspects of machine learning model serialization formats. This list covers vulnerabilities, attack vectors, detection tools, and mitigation strategies for various serialization formats used in machine learning.

## Contents

- [Serialization Formats and Vulnerabilities](#serialization-formats-and-vulnerabilities)
  - [Pickle](#pickle)
  - [PyTorch](#pytorch)
  - [Joblib](#joblib)
  - [Keras/H5](#kerash5)
  - [GGUF](#gguf)
  - [ONNX](#onnx)
  - [SafeTensors](#safetensors)
  - [TFLite](#tflite)
  - [TorchScript](#torchscript)
  - [NeMo](#nemo)
- [Advanced Attack Techniques](#advanced-attack-techniques)
  - [Hydra/OmegaConf Configuration Injection](#hydraomegaconf-configuration-injection)
  - [Format Downgrade Attacks](#format-downgrade-attacks)
  - [Neural Network Steganography](#neural-network-steganography)
  - [Neural Trojans and Backdoors](#neural-trojans-and-backdoors)
  - [Scanner Bypass Techniques](#scanner-bypass-techniques)
- [Detection and Scanning Tools](#detection-and-scanning-tools)
  - [Open Source Tools](#open-source-tools)
- [Supply Chain Attacks](#supply-chain-attacks)
  - [Academic Research](#academic-research)
  - [Industry Reports](#industry-reports)
  - [Case Studies](#case-studies)
  - [Academic Institutions](#academic-institutions)
  - [Individual Researchers](#individual-researchers)
- [Mitigation Strategies](#mitigation-strategies)
- [Emerging Research Directions](#emerging-research-directions)
- [Contributing](#contributing)

## Serialization Formats and Vulnerabilities

### Pickle

Pickle is Python's native serialization format, widely used in ML frameworks but notorious for its security vulnerabilities.

#### Vulnerabilities

- **Arbitrary Code Execution**: Pickle allows arbitrary code execution during deserialization, making it vulnerable to remote code execution attacks.
- **Supply Chain Risks**: Models distributed via public repositories can contain malicious code that executes upon loading.
- **Credential Theft**: Malicious pickle files can steal cloud credentials, SSH keys, and other sensitive information.

#### Research and Documentation

- [ProtectAI Knowledge Base: Pickle Model Arbitrary Code Execution](https://protectai.com/insights/knowledge-base/deserialization-threats/PAIT-PKL-100) - Detailed explanation of how pickle-based attacks work and their potential impact.
- [Python Pickle Poisoning and Backdooring Pth Files](https://snyk.io/articles/python-pickle-poisoning-and-backdooring-pth-files/) - Analysis of pickle vulnerabilities in PyTorch context by Snyk.
- [Never a dill moment: Exploiting machine learning pickle files](https://blog.trailofbits.com/2021/11/18/never-a-dill-moment-exploiting-machine-learning-pickle-files/) - Comprehensive breakdown of pickle exploitation techniques by Trail of Bits.
- [Pickle Serialization in Data Science: A Ticking Time Bomb](https://www.robustintelligence.com/blog-posts/pickle-serialization-in-data-science-a-ticking-time-bomb) - Analysis by Robust Intelligence on the dangers of pickle in data science workflows.
- [Enhancing cloud security in AI/ML: The little pickle story](https://aws.amazon.com/blogs/security/enhancing-cloud-security-in-ai-ml-the-little-pickle-story/) - AWS security team's analysis of pickle vulnerabilities in cloud environments.
- [Paws in the Pickle Jar: Risk & Vulnerability in the Model-Sharing Ecosystem](https://www.splunk.com/en_us/blog/security/paws-in-the-pickle-jar-risk-vulnerability-in-the-model-sharing-ecosystem.html) - Splunk's research on pickle vulnerabilities in shared ML models.

### PyTorch

PyTorch uses pickle for serialization by default, inheriting its vulnerabilities while adding some format-specific issues.

#### Vulnerabilities

- **Multiple File Formats**: PyTorch has multiple serialization formats with different security properties, causing confusion and misuse.
- **Polyglot Files**: PyTorch files can be polyglots (valid in multiple formats simultaneously), enabling bypass of security controls.
- **Stacked Pickle Files**: Some PyTorch formats use stacked pickle files, complicating security analysis.
- **weights_only Bypass (CVE-2025-32434)**: Critical RCE vulnerability allowing code execution even with `weights_only=True` parameter.

#### Research and Documentation

- [Shelltorch Explained: Multiple Vulnerabilities in Pytorch Model Server](https://www.oligo.security/blog/shelltorch-explained-multiple-vulnerabilities-in-pytorch-model-server) - Analysis of critical vulnerabilities in PyTorch model serving by Oligo Security.
- [pickle is a security issue #52596 - pytorch](https://github.com/pytorch/pytorch/issues/52596) - Discussion of pickle security issues in PyTorch.
- [Unpickling Pytorch: Keeping Malicious AI Out](https://www.sonatype.com/resources/whitepapers/unpickling-pytorch) - Whitepaper on PyTorch security risks and mitigations by Sonatype.
- [Dear PyTorch, Nobody Likes Pickles](https://aikomail.com/blog/dear-pytorch-nobody-likes-pickles) - Analysis of PyTorch's reliance on pickle and alternatives.
- [Multiple deserialization vulnerabilities in PyTorch Lightning](https://www.kb.cert.org/vuls/id/252619) - CERT advisory on PyTorch Lightning vulnerabilities.
- [Zero-Trust Artificial Intelligence Model Security](https://arxiv.org/html/2503.01758v1) - Research on securing PyTorch file formats.
- [CVE-2025-32434: torch.load with weights_only=True RCE](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6) - Critical vulnerability in PyTorch's "safe" loading mode.
- [Wiz: CVE-2025-32434 Analysis](https://www.wiz.io/vulnerability-database/cve/cve-2025-32434) - Technical analysis of the weights_only bypass.
- [Kaspersky: Critical vulnerability in PyTorch framework](https://www.kaspersky.com/blog/vulnerability-in-pytorch-framework/53311/) - Overview of CVE-2025-32434 impact.

### Joblib

Joblib is commonly used for serializing scikit-learn models but relies on pickle for non-numerical data.

#### Vulnerabilities

- **Pickle Dependency**: Joblib uses pickle for serializing Python objects, inheriting its security issues.
- **Mixed Security Model**: Optimized for numerical arrays but falls back to pickle for other objects, creating a false sense of security.

#### Research and Documentation

- [ProtectAI Knowledge Base: Joblib Model Arbitrary Code Execution](https://protectai.com/insights/knowledge-base/deserialization-threats/PAIT-JOBLIB-100) - Detailed explanation of Joblib vulnerabilities.
- [Snyk Vulnerability: SNYK-PYTHON-JOBLIB-6913425](https://security.snyk.io/vuln/SNYK-PYTHON-JOBLIB-6913425) - Security advisory for Joblib vulnerabilities by Snyk.
- [Abusing ML model file formats to create malware on AI systems](https://github.com/Azure/counterfit/wiki/Abusing-ML-model-file-formats-to-create-malware-on-AI-systems:-A-proof-of-concept) - Microsoft's research on exploiting Joblib serialization.

### Keras/H5

Keras models can be serialized in H5 format, which has its own security concerns.

#### Vulnerabilities

- **Lambda Layer Execution**: Keras models with Lambda layers can execute arbitrary code when loaded.
- **Custom Objects**: Loading models with custom objects can lead to code execution.
- **safe_mode Bypass via Downgrade**: The `safe_mode=True` parameter is ignored when loading legacy H5 format checkpoints (CVE-2024-3660 bypass).

#### Research and Documentation

- [ProtectAI Knowledge Base: Keras Model Lambda Layer Arbitrary Code Execution](https://protectai.com/insights/knowledge-base/deserialization-threats/PAIT-KERAS-101) - Detailed explanation of Keras Lambda layer vulnerabilities.
- [How to Hunt Vulnerabilities in Machine Learning Model File Formats](https://blog.huntr.com/hunting-vulnerabilities-in-machine-learning-model-file-formats) - Section on Keras Lambda layers exploitation.
- [Researchers Uncover Flaws in Popular Open-Source ML Tools](https://thehackernews.com/2024/12/researchers-uncover-flaws-in-popular.html) - The Hacker News coverage of vulnerabilities in H5 format.
- [TensorFlow Keras Downgrade Attack: CVE-2024-3660 Bypass](https://www.oligo.security/blog/tensorflow-keras-downgrade-attack-cve-2024-3660-bypass) - Analysis of how safe_mode can be bypassed using legacy H5 formats.
- [Exposing Keras Lambda Exploits in TensorFlow Models](https://blog.huntr.com/exposing-keras-lambda-exploits-in-tensorflow-models) - Detailed exploitation guide for Lambda layers.

### GGUF

GGUF (GPT-Generated Unified Format) is used for large language models, particularly with llama.cpp.

#### Vulnerabilities

- **Jinja Template Injection**: GGUF models can contain malicious Jinja templates that execute code when rendered (CVE-2024-34359).
- **Metadata Exploitation**: Attackers can embed malicious code in model metadata.
- **Memory Corruption**: Integer overflow vulnerabilities in GGUF parsers can lead to heap corruption and code execution.

#### Research and Documentation

- [ProtectAI Knowledge Base: GGUF Model Template Containing Arbitrary Code Execution](https://protectai.com/insights/knowledge-base/deserialization-threats/PAIT-GGUF-101) - Detailed explanation of GGUF vulnerabilities.
- [CVE-2024-34359](https://nvd.nist.gov/vuln/detail/CVE-2024-34359) - Vulnerability in GGUF format related to Jinja templates.
- [Beware of Exploitable ML Clients and "Safe" Models](https://jfrog.com/blog/machine-learning-bug-bonanza-exploiting-ml-clients-and-safe-models/) - JFrog's research on GGUF vulnerabilities.
- [GGML GGUF File Format Vulnerabilities](https://www.databricks.com/blog/ggml-gguf-file-format-vulnerabilities) - Databricks analysis of memory corruption bugs in GGUF parsers.
- [Cisco Talos: TALOS-2024-1913](https://talosintelligence.com/vulnerability_reports/TALOS-2024-1913) - Heap overflow vulnerabilities in GGUF.
- [GGUF File Format Vulnerabilities: A Guide for Hackers](https://blog.huntr.com/gguf-file-format-vulnerabilities-a-guide-for-hackers) - Comprehensive vulnerability hunting guide.
- [JFrog: GGUF-SSTI Research](https://research.jfrog.com/model-threats/gguf-ssti/) - Server-side template injection analysis.
- [HuggingFace GGUF Jinja Analysis](https://github.com/huggingface/gguf-jinja-analysis) - Large-scale analysis of GGUF chat templates for malicious code.

### ONNX

ONNX (Open Neural Network Exchange) is designed for model interoperability but has security considerations.

#### Vulnerabilities

- **Custom Operators**: ONNX supports custom operators that can execute arbitrary code.
- **Complex Control Flow**: Models with complex branching logic can lead to unexpected execution paths.
- **Native Code Integration**: Custom operators can be implemented in native code for "performance," creating security risks.

#### Research and Documentation

- [How to Hunt Vulnerabilities in Machine Learning Model File Formats: ONNX Custom Operators](https://blog.huntr.com/hunting-vulnerabilities-in-machine-learning-model-file-formats) - Section on ONNX custom operator vulnerabilities.
- [Building Secure AI Systems with Union's Defense-in-Depth Approach](https://www.union.ai/blog-post/building-secure-ai-systems-with-unions-defense-in-depth-approach) - Discussion of ONNX security considerations.
- [Securely serializing/loading untrusted pytorch models](https://discuss.pytorch.org/t/securely-serializing-loading-untrusted-pytorch-models/119744) - Discussion of ONNX as a safer alternative to pickle.

### SafeTensors

SafeTensors was created specifically to address security issues in other serialization formats, particularly pickle.

#### Vulnerabilities

- **Conversion Service Attacks**: The Hugging Face SafeTensors conversion service has been vulnerable to attacks.
- **Implementation Flaws**: While the format itself is designed for security, implementation issues can still arise.
- **Metadata-Based Attacks**: SafeTensors + Hydra configs = Still vulnerable via config.json injection.

#### Research and Documentation

- [Hijacking Safetensors Conversion on Hugging Face](https://hiddenlayer.com/innovation-hub/silent-sabotage/) - Detailed analysis of vulnerabilities in the SafeTensors conversion service by HiddenLayer.
- [EleutherAI SafeTensors - Final Report](https://huggingface.co/datasets/safetensors/trail_of_bits_audit_repot/resolve/main/SOW-TrailofBits-EleutherAI_HuggingFace-v1.2.pdf) - Security audit of the SafeTensors library by Trail of Bits.
- [A Secure Alternative to Pickle for ML Models](https://dev.to/lukehinds/understanding-safetensors-a-secure-alternative-to-pickle-for-ml-models-o71) - Overview of SafeTensors security benefits.
- [New Hugging Face Vulnerability Exposes AI Models to Backdoors](https://thehackernews.com/2024/02/new-hugging-face-vulnerability-exposes.html) - The Hacker News coverage of SafeTensors conversion vulnerabilities.
- [Unit 42: RCE With Modern AI/ML Formats](https://unit42.paloaltonetworks.com/rce-vulnerabilities-in-ai-python-libraries/) - How SafeTensors can still be exploited via Hydra configuration injection.

### TFLite

TensorFlow Lite (TFLite) is a lightweight solution for mobile and edge devices.

#### Vulnerabilities

- **Interpreter Vulnerabilities**: The TFLite interpreter can be vulnerable to memory corruption issues.
- **Malicious Operations**: Custom operations can be exploited for code execution.
- **Model Tampering**: TFLite models can be modified to include malicious code.
- **H5 Conversion Vulnerabilities**: TFLite Converter executes code when converting malicious H5 models.

#### Research and Documentation

- [Unraveling the Characterization and Propagation of Security Vulnerabilities in TensorFlow](https://conf.researchr.org/details/internetware-2025/internetware-2025-research-track/44/Unraveling-the-Characterization-and-Propagation-of-Security-Vulnerabilities-in-Tensor) - Research on TensorFlow and TFLite vulnerabilities.
- [Deserialization bug in TensorFlow machine learning framework allowed arbitrary code execution](https://portswigger.net/daily-swig/deserialization-bug-in-tensorflow-machine-learning-framework-allowed-arbitrary-code-execution) - Analysis of TensorFlow deserialization vulnerabilities affecting TFLite.
- [TensorFlow Keras Downgrade Attack](https://www.oligo.security/blog/tensorflow-keras-downgrade-attack-cve-2024-3660-bypass) - TFLite Converter as attack vector.

### TorchScript

TorchScript is a way to create serializable and optimizable models from PyTorch code.

#### Vulnerabilities

- **Pickle Dependency**: Some TorchScript formats include pickle files, inheriting their vulnerabilities.
- **Multiple Format Versions**: TorchScript has multiple versions with different security properties.
- **Code Execution**: TorchScript can execute arbitrary code during loading.

#### Research and Documentation

- [Relishing new Fickling features for securing ML systems](https://blog.trailofbits.com/2024/03/04/relishing-new-fickling-features-for-securing-ml-systems/) - Trail of Bits' analysis of TorchScript security issues.
- [Machine Learning Models are Code](https://hiddenlayer.com/innovation-hub/models-are-code/) - HiddenLayer's research on TorchScript vulnerabilities.

### NeMo

NVIDIA NeMo is a PyTorch-based framework for building AI models, using its own `.nemo` format.

#### Vulnerabilities

- **Hydra Configuration Injection (CVE-2025-23304)**: NeMo uses Hydra for configuration management, allowing RCE through crafted metadata.
- **Unsafe Instantiation**: The `hydra.utils.instantiate()` function accepts any callable, not just class names.

#### Research and Documentation

- [CVE-2025-23304](https://nvidia.custhelp.com/app/answers/detail/a_id/5718) - NVIDIA Security Bulletin for NeMo Framework.
- [Unit 42: Remote Code Execution With Modern AI/ML Formats](https://unit42.paloaltonetworks.com/rce-vulnerabilities-in-ai-python-libraries/) - Discovery of Hydra-based vulnerabilities in NeMo.
- [The Register: Python libraries in AI/ML models can be poisoned](https://www.theregister.com/2026/01/13/ai_python_library_bugs_allow/) - Coverage of NeMo, Uni2TS, and FlexTok vulnerabilities.

### TensorFlow SavedModel

TensorFlow's default serialization format, often considered "safe" but with significant abuse potential.

#### Vulnerabilities

- **tf.io Operations Abuse**: SavedModel can contain tf.io.read_file/write_file operations for arbitrary file system access.
- **Data Exfiltration**: Models can read sensitive files and return content via inference API.
- **Webshell Deployment**: Models can write malicious files to web directories.

#### Research and Documentation

- [HiddenLayer: Machine Learning Models are Code](https://hiddenlayer.com/innovation-hub/models-are-code/) - Comprehensive analysis of SavedModel abuse via tf.io operations.
- [TensorFlow Security Policy](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md) - Official security documentation (notes tf.io risks as "by design").
- [Deep Dive into the Abuse of DL APIs](https://arxiv.org/pdf/2601.04553) - Academic research on exploiting TensorFlow APIs.

#### Research and Documentation

- [Max severity RCE flaw discovered in widely used Apache Parquet](https://news.ycombinator.com/item?id=43603091) - Discussion of vulnerabilities in Parquet format.
- [Save Time and Money Using Parquet and Feather in Python](https://medium.com/codex/save-time-and-money-using-parquet-and-feather-in-python-d5d6c0b93899) - Discussion of Parquet and Feather formats with security implications.
- [Deserialization bug in TensorFlow machine learning framework allowed arbitrary code execution](https://portswigger.net/daily-swig/deserialization-bug-in-tensorflow-machine-learning-framework-allowed-arbitrary-code-execution) - Analysis of YAML deserialization vulnerabilities in TensorFlow.

---

## Advanced Attack Techniques

### Hydra/OmegaConf Configuration Injection

A new class of attacks targeting ML configuration management systems, allowing RCE even with "safe" formats like SafeTensors.

#### Overview

Many modern ML libraries use [Hydra](https://github.com/facebookresearch/hydra) (maintained by Meta) for configuration management. The `hydra.utils.instantiate()` function accepts any callable, enabling code execution through crafted model metadata.

#### Affected Libraries

| Library | Vendor | CVE | Fixed Version |
|---------|--------|-----|---------------|
| NeMo | NVIDIA | CVE-2025-23304 | 2.3.2 |
| Uni2TS | Salesforce | CVE-2026-22584 | July 2025 |
| FlexTok | Apple/EPFL | N/A | June 2025 |

#### Research and Documentation

- [Unit 42: Remote Code Execution With Modern AI/ML Formats](https://unit42.paloaltonetworks.com/rce-vulnerabilities-in-ai-python-libraries/) - Original research discovering these vulnerabilities.
- [The Register: AI Python library bugs allow RCE](https://www.theregister.com/2026/01/13/ai_python_library_bugs_allow/) - Industry coverage.

### Format Downgrade Attacks

Attacks exploiting backward compatibility to bypass security fixes by loading models in legacy formats.

#### Keras H5 Downgrade

The `safe_mode=True` parameter is completely ignored when loading old H5 format checkpoints, allowing CVE-2024-3660 bypass.

#### Research and Documentation

- [Oligo Security: TensorFlow Keras Downgrade Attack](https://www.oligo.security/blog/tensorflow-keras-downgrade-attack-cve-2024-3660-bypass) - Detailed analysis of the downgrade attack.

### Neural Network Steganography

Techniques for hiding malware inside model weights rather than in serialization code.

#### Methods

| Method | Paper | Embedding Rate | Performance Impact |
|--------|-------|----------------|-------------------|
| LSB Substitution | StegoNet | ~15% | High |
| MSB Reservation | EvilModel 2.0 | ~48.52% | None |
| Fast Substitution | EvilModel 2.0 | ~48.52% | None |
| Half Substitution | EvilModel 2.0 | ~48.52% | None |
| Spread-Spectrum | MaleficNet | Variable | Low |

#### Research and Documentation

- [arXiv: EvilModel - Hiding Malware Inside Neural Networks](https://arxiv.org/abs/2107.08590) - Original EvilModel paper.
- [arXiv: EvilModel 2.0](https://arxiv.org/abs/2109.04344) - Improved techniques with 48.52% embedding rate.
- [ScienceDirect: EvilModel 2.0 (Peer-reviewed)](https://www.sciencedirect.com/science/article/abs/pii/S0167404822002012) - Published version.
- [ESORICS 2022: MaleficNet](https://doi.org/10.1007/978-3-031-17143-7_21) - Spread-spectrum channel coding approach.
- [arXiv: NeuPerm - Disrupting Malware in Neural Networks](https://arxiv.org/pdf/2510.20367) - Defense mechanism using layer permutations.

### Neural Trojans and Backdoors

Behavioral backdoors that activate during inference when specific triggers appear in input data.

#### Attack Types

| Type | Description |
|------|-------------|
| BadNets | Pixel pattern triggers misclassification |
| Trojan Attack | Specific input features activate backdoor |
| Latent Backdoor | Inherited from teacher model in transfer learning |
| Clean Label | No data modification, uses adversarial perturbations |

#### Research and Documentation

- [IEEE S&P: Neural Cleanse](https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf) - Foundational backdoor detection work.
- [arXiv: Detecting AI Trojans Using Meta Neural Analysis](https://arxiv.org/pdf/1910.03137) - MNTD framework.
- [NIST TrojAI Program](https://pages.nist.gov/trojai/) - Official NIST trojan detection challenges.
- [GitHub: TrojAI Literature](https://github.com/usnistgov/trojai-literature) - Comprehensive literature repository.
- [arXiv: Survey on Neural Trojans](https://eprint.iacr.org/2020/201.pdf) - Academic survey.
- [arXiv: Linear Weight Classification for Trojan Detection](https://arxiv.org/html/2411.03445v1) - Weight-based detection approach.

### Scanner Bypass Techniques

Methods for evading current model security scanners.

#### Techniques

| Technique | Bypasses | Mechanism |
|-----------|----------|-----------|
| Wrapper Functions | picklescan, modelscan | Use allowed modules that internally call dangerous functions |
| Dill/Cloudpickle | picklescan, modelscan | Serialize lambdas with obfuscated code |
| Indirect Invocation | All static scanners | Call dangerous functions via getattr chains |
| Polyglot Files | Format-specific scanners | Valid in multiple formats, exploit parser confusion |
| Legacy Format Loading | Safe-mode checks | Trigger code paths without security checks |
| Weight Steganography | All current scanners | Hide payload in weights, not code |
| Config Injection | All current scanners | Payload in YAML/JSON metadata |

#### Research and Documentation

- [CCS'25: PickleBall - Secure Pickle Deserialization](https://cs.brown.edu/~vpk/papers/pickleball.ccs25.pdf) - Policy-based defense approach.
- [Sonatype: Bypassing picklescan](https://www.sonatype.com/blog/bypassing-picklescan-sonatype-discovers-four-vulnerabilities) - Four bypass vulnerabilities discovered.

---

## Detection and Scanning Tools

### Open Source Tools

#### ModelScan

ModelScan is an open-source tool from Protect AI that scans models to determine if they contain unsafe code.

- **Supported Formats**: H5, Pickle, SavedModel, and more
- **Features**: Fast scanning, severity ranking, CLI and API interfaces
- **GitHub**: [protectai/modelscan](https://github.com/protectai/modelscan)
- **Documentation**: [ModelScan Documentation](https://protectai.com/modelscan)

#### Fickling

Fickling is a decompiler, static analyzer, and bytecode rewriter for the Python pickle module, developed by Trail of Bits.

- **Supported Formats**: Pickle, PyTorch
- **Features**: Decompilation, static analysis, injection capabilities, polyglot detection
- **GitHub**: [trailofbits/fickling](https://github.com/trailofbits/fickling)
- **Documentation**: [Relishing new Fickling features for securing ML systems](https://blog.trailofbits.com/2024/03/04/relishing-new-fickling-features-for-securing-ml-systems/)

#### PickleScan

PickleScan is a security scanner specifically designed for Python pickle files, developed by Hugging Face.

- **Supported Formats**: Pickle
- **Features**: Detection of malicious patterns, integration with CI/CD
- **GitHub**: [mmaitre314/picklescan](https://github.com/mmaitre314/picklescan)
- **Documentation**: [Bypassing picklescan: Sonatype discovers four vulnerabilities](https://www.sonatype.com/blog/bypassing-picklescan-sonatype-discovers-four-vulnerabilities)
- **Hugging Face Documentation**: [Pickle Scanning](https://huggingface.co/docs/hub/en/security-pickle)

#### ModelAudit

ModelAudit is a comprehensive scanner from promptfoo with the widest format coverage.

- **Supported Formats**: PyTorch, TensorFlow, Keras, ONNX, SafeTensors, GGUF/GGML, Flax/JAX, Pickle, Joblib, NumPy, PMML, and more
- **Features**: 29 specialized scanners, weight anomaly detection, secrets detection, network analysis, SBOM generation
- **PyPI**: [modelaudit](https://pypi.org/project/modelaudit/)
- **Documentation**: [ModelAudit Scanners](https://www.promptfoo.dev/docs/model-audit/scanners/)

### Scanner Comparison

| Scanner | Pickle | PyTorch | Keras/H5 | GGUF | SafeTensors | Hydra Configs | Weight Steganography |
|---------|--------|---------|----------|------|-------------|---------------|---------------------|
| picklescan | ✅ | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ |
| modelscan | ✅ | ⚠️ | ✅ | ❌ | ❌ | ❌ | ❌ |
| fickling | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| ModelAudit | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ⚠️ |

---

## Supply Chain Attacks

### Academic Research

- [A Large-Scale Exploit Instrumentation Study of AI/ML Supply Chain Attacks in Hugging Face Models](https://arxiv.org/abs/2410.04490) - Comprehensive study of serialization vulnerabilities across Hugging Face by researchers from University of Notre Dame and University of Hawaii.
- [Supply-Chain Attacks in Machine Learning Frameworks](https://openreview.net/attachment?id=EH5PZW6aCr&name=pdf) - Analysis of ML supply chain security beyond algorithmic-level attacks.
- [Securing Machine Learning Supply Chains](https://apps.dtic.mil/sti/trecms/pdf/AD1164482.pdf) - Research on ML supply chain vulnerabilities and mitigations.
- [Towards Measuring Malicious Code Poisoning Attacks on Pre-trained Model Hubs](https://arxiv.org/html/2409.09368v1) - First systematic study of malicious code poisoning attacks on pre-trained model hubs.
- [Zero-Trust Artificial Intelligence Model Security](https://arxiv.org/html/2503.01758v1) - Research on securing AI models against supply chain attacks.
- [CCS'25: PickleBall - Secure Deserialization of Pickle-based ML Models](https://cs.brown.edu/~vpk/papers/pickleball.ccs25.pdf) - Policy-based runtime protection for pickle deserialization.

### Industry Reports

- [Malicious AI Models Undermine Software Supply-Chain Security](https://cacm.acm.org/research/malicious-ai-models-undermine-software-supply-chain-security/) - ACM report on emerging threats from malicious AI models.
- [AI Supply Chain Security: Hugging Face Malicious ML Models](https://nsfocusglobal.com/ai-supply-chain-security-hugging-face-malicious-ml-models/) - Analysis of malicious models on Hugging Face by NSFocus.
- [Hugging Face partners with Wiz Research to Improve AI Security](https://huggingface.co/blog/hugging-face-wiz-security-blog) - Collaborative efforts to improve AI model security.
- [The 2025 Software Supply Chain Security Report](https://ntsc.org/wp-content/uploads/2025/03/The-2025-Software-Supply-Chain-Security-Report-RL-compressed.pdf) - ReversingLabs' report on software supply chain security, including ML models.
- [Hugging Face Vulnerabilities Highlight AI-as-a-Service Risks](https://www.bankinfosecurity.com/hugging-face-vulnerabilities-highlight-ai-as-a-service-risks-a-24807) - Analysis of vulnerabilities in AI-as-a-Service platforms.
- [Unit 42: RCE Vulnerabilities in AI/ML Python Libraries](https://unit42.paloaltonetworks.com/rce-vulnerabilities-in-ai-python-libraries/) - January 2026 research on Hydra-based attacks.

### Case Studies

- [Examining Malicious Hugging Face ML Models with Silent Backdoor](https://jfrog.com/blog/data-scientists-targeted-by-malicious-hugging-face-ml-models-with-silent-backdoor/) - Analysis of real-world malicious models by JFrog.
- [Malicious ML models discovered on Hugging Face platform](https://www.reversinglabs.com/blog/rl-identifies-malware-ml-model-hosted-on-hugging-face) - Discovery of flaws in the Picklescan security feature by ReversingLabs.
- [Weaponizing ML Models with Ransomware](https://hiddenlayer.com/innovation-hub/weaponizing-machine-learning-models-with-ransomware/) - Demonstration of ransomware delivery through ML models by HiddenLayer.
- [Hugging Face platform continues to be plagued by vulnerable pickles](https://cyberscoop.com/hugging-face-platform-continues-to-be-plagued-by-vulnerable-pickles/) - CyberScoop's coverage of ongoing pickle vulnerabilities.
- [Wiz researchers hacked into leading AI infrastructure providers](https://www.techtarget.com/searchsecurity/news/366602365/Wiz-researchers-hacked-into-leading-AI-infrastructure-providers) - Wiz's research on vulnerabilities in AI infrastructure providers.

### Academic Institutions

#### Harvard University

- **Focus**: Robust machine learning
- **Key Research**: [Understanding AI Vulnerabilities](https://www.harvardmagazine.com/2025/03/artificial-intelligence-vulnerabilities-harvard-yaron-singer)

#### Brown University

- **Focus**: Secure deserialization
- **Key Research**: [PickleBall - Secure Pickle Deserialization](https://cs.brown.edu/~vpk/papers/pickleball.ccs25.pdf)

### Individual Researchers

#### Beatrice Casey (University of Notre Dame)

- **Focus**: ML supply chain security
- **Key Research**: [A Large-Scale Exploit Instrumentation Study of AI/ML Supply Chain Attacks in Hugging Face Models](https://arxiv.org/abs/2410.04490)

#### Ji'an Zhou

- **Focus**: PyTorch security
- **Key Research**: CVE-2025-32434 (torch.load weights_only bypass)

---

## Mitigation Strategies

### Format Selection

- **Use Safer Formats**: Prefer formats like SafeTensors or ONNX over pickle-based formats when possible.
- **Avoid Legacy Formats**: Don't load old H5 checkpoints; convert to new Keras format.
- **Format Conversion**: Convert models to safer formats before deployment.

### Scanning and Detection

- **Implement Scanning**: Integrate model scanning into CI/CD pipelines and before loading any model.
- **Multi-Scanner Approach**: Use multiple scanners (ModelAudit + Fickling) to maximize coverage.
- **Config Scanning**: Scan YAML/JSON configs for Hydra injection patterns (currently manual).

### Runtime Protection

- **Sandboxing**: Run model loading in isolated environments with limited permissions.
- **Syscall Filtering**: Use seccomp/AppArmor to restrict model loading operations.
- **Policy-Based Deserialization**: Consider PickleBall approach for pickle-based models.

### Verification and Provenance

- **Verification**: Implement cryptographic verification of model origins.
- **Model Signing**: Require signed model artifacts and verify before loading.
- **Trusted Sources**: Only load models from verified, trusted sources.

### Code and Dependency Management

- **Code Review**: Review custom operators and Lambda layers for security issues.
- **Dependency Management**: Keep ML libraries and dependencies updated (especially PyTorch ≥2.6.0).
- **Version Pinning**: Pin known-safe versions of ML frameworks.

### Operational Security

- **Input Validation**: Validate all inputs to ML models.
- **Principle of Least Privilege**: Run ML systems with minimal required permissions.
- **Monitoring**: Monitor ML systems for unusual behavior (file access, network connections).
- **Network Isolation**: Restrict network access for model loading processes.

---

## Emerging Research Directions

### Areas Needing Research

- **Hydra Configuration Scanner**: No current tool validates Hydra `_target_` values for malicious patterns.
- **Weight Steganography Detection**: Limited detection capabilities for malware hidden in model weights.
- **Format Downgrade Matrix**: Comprehensive mapping of backward compatibility as attack vectors.
- **Trojan Detection Integration**: Behavioral backdoor detection in model scanning pipelines.
- **GGUF Fuzzing**: Systematic vulnerability discovery in GGUF/GGML parsers.

### Promising Approaches

- **LLM-Based Semantic Detection**: Using LLMs to understand dangerous API chains.
- **Unified Policy-Based Deserialization**: Runtime allow-lists based on library analysis (PickleBall).
- **Format-Agnostic Behavioral Sandboxing**: gVisor/Firecracker/WASM isolation.
- **Automated Vulnerability Discovery**: AFL++/libFuzzer for ML format parsers.

---

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to suggest additions or improvements to this list.

### Contribution Guidelines

- Include relevant CVE numbers where applicable
- Provide links to original research/advisories
- Note scanner coverage gaps for new vulnerabilities
- Include PoC references (where publicly available)

## License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
