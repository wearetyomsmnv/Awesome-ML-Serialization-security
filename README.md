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
- [Detection and Scanning Tools](#detection-and-scanning-tools)
  - [Open Source Tools](#open-source-tools)
- [Supply Chain Attacks](#supply-chain-attacks)
  - [Academic Research](#academic-research)
  - [Industry Reports](#industry-reports)
  - [Case Studies](#case-studies)
  - [Academic Institutions](#academic-institutions)
  - [Individual Researchers](#individual-researchers)
- [Mitigation Strategies](#mitigation-strategies)
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

#### Research and Documentation

- [Shelltorch Explained: Multiple Vulnerabilities in Pytorch Model Server](https://www.oligo.security/blog/shelltorch-explained-multiple-vulnerabilities-in-pytorch-model-server) - Analysis of critical vulnerabilities in PyTorch model serving by Oligo Security.
- [pickle is a security issue #52596 - pytorch](https://github.com/pytorch/pytorch/issues/52596) - Discussion of pickle security issues in PyTorch.
- [Unpickling Pytorch: Keeping Malicious AI Out](https://www.sonatype.com/resources/whitepapers/unpickling-pytorch) - Whitepaper on PyTorch security risks and mitigations by Sonatype.
- [Dear PyTorch, Nobody Likes Pickles](https://aikomail.com/blog/dear-pytorch-nobody-likes-pickles) - Analysis of PyTorch's reliance on pickle and alternatives.
- [Multiple deserialization vulnerabilities in PyTorch Lightning](https://www.kb.cert.org/vuls/id/252619) - CERT advisory on PyTorch Lightning vulnerabilities.
- [Zero-Trust Artificial Intelligence Model Security](https://arxiv.org/html/2503.01758v1) - Research on securing PyTorch file formats.

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

#### Research and Documentation

- [ProtectAI Knowledge Base: Keras Model Lambda Layer Arbitrary Code Execution](https://protectai.com/insights/knowledge-base/deserialization-threats/PAIT-KERAS-101) - Detailed explanation of Keras Lambda layer vulnerabilities.
- [How to Hunt Vulnerabilities in Machine Learning Model File Formats](https://blog.huntr.com/hunting-vulnerabilities-in-machine-learning-model-file-formats) - Section on Keras Lambda layers exploitation.
- [Researchers Uncover Flaws in Popular Open-Source ML Tools](https://thehackernews.com/2024/12/researchers-uncover-flaws-in-popular.html) - The Hacker News coverage of vulnerabilities in H5 format.

### GGUF

GGUF (GPT-Generated Unified Format) is used for large language models, particularly with llama.cpp.

#### Vulnerabilities

- **Jinja Template Injection**: GGUF models can contain malicious Jinja templates that execute code when rendered.
- **Metadata Exploitation**: Attackers can embed malicious code in model metadata.

#### Research and Documentation

- [ProtectAI Knowledge Base: GGUF Model Template Containing Arbitrary Code Execution](https://protectai.com/insights/knowledge-base/deserialization-threats/PAIT-GGUF-101) - Detailed explanation of GGUF vulnerabilities.
- [CVE-2024-34359](https://nvd.nist.gov/vuln/detail/CVE-2024-34359) - Vulnerability in GGUF format related to Jinja templates.
- [Beware of Exploitable ML Clients and "Safe" Models](https://jfrog.com/blog/machine-learning-bug-bonanza-exploiting-ml-clients-and-safe-models/) - JFrog's research on GGUF vulnerabilities.

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

#### Research and Documentation

- [Hijacking Safetensors Conversion on Hugging Face](https://hiddenlayer.com/innovation-hub/silent-sabotage/) - Detailed analysis of vulnerabilities in the SafeTensors conversion service by HiddenLayer.
- [EleutherAI SafeTensors - Final Report](https://huggingface.co/datasets/safetensors/trail_of_bits_audit_repot/resolve/main/SOW-TrailofBits-EleutherAI_HuggingFace-v1.2.pdf) - Security audit of the SafeTensors library by Trail of Bits.
- [A Secure Alternative to Pickle for ML Models](https://dev.to/lukehinds/understanding-safetensors-a-secure-alternative-to-pickle-for-ml-models-o71) - Overview of SafeTensors security benefits.
- [New Hugging Face Vulnerability Exposes AI Models to Backdoors](https://thehackernews.com/2024/02/new-hugging-face-vulnerability-exposes.html) - The Hacker News coverage of SafeTensors conversion vulnerabilities.

### TFLite

TensorFlow Lite (TFLite) is a lightweight solution for mobile and edge devices.

#### Vulnerabilities

- **Interpreter Vulnerabilities**: The TFLite interpreter can be vulnerable to memory corruption issues.
- **Malicious Operations**: Custom operations can be exploited for code execution.
- **Model Tampering**: TFLite models can be modified to include malicious code.

#### Research and Documentation

- [Unraveling the Characterization and Propagation of Security Vulnerabilities in TensorFlow](https://conf.researchr.org/details/internetware-2025/internetware-2025-research-track/44/Unraveling-the-Characterization-and-Propagation-of-Security-Vulnerabilities-in-Tensor) - Research on TensorFlow and TFLite vulnerabilities.
- [Deserialization bug in TensorFlow machine learning framework allowed arbitrary code execution](https://portswigger.net/daily-swig/deserialization-bug-in-tensorflow-machine-learning-framework-allowed-arbitrary-code-execution) - Analysis of TensorFlow deserialization vulnerabilities affecting TFLite.

### TorchScript

TorchScript is a way to create serializable and optimizable models from PyTorch code.

#### Vulnerabilities

- **Pickle Dependency**: Some TorchScript formats include pickle files, inheriting their vulnerabilities.
- **Multiple Format Versions**: TorchScript has multiple versions with different security properties.
- **Code Execution**: TorchScript can execute arbitrary code during loading.

#### Research and Documentation

- [Relishing new Fickling features for securing ML systems](https://blog.trailofbits.com/2024/03/04/relishing-new-fickling-features-for-securing-ml-systems/) - Trail of Bits' analysis of TorchScript security issues.
- [Machine Learning Models are Code](https://hiddenlayer.com/innovation-hub/models-are-code/) - HiddenLayer's research on TorchScript vulnerabilities.



#### Research and Documentation

- [Max severity RCE flaw discovered in widely used Apache Parquet](https://news.ycombinator.com/item?id=43603091) - Discussion of vulnerabilities in Parquet format.
- [Save Time and Money Using Parquet and Feather in Python](https://medium.com/codex/save-time-and-money-using-parquet-and-feather-in-python-d5d6c0b93899) - Discussion of Parquet and Feather formats with security implications.
- [Deserialization bug in TensorFlow machine learning framework allowed arbitrary code execution](https://portswigger.net/daily-swig/deserialization-bug-in-tensorflow-machine-learning-framework-allowed-arbitrary-code-execution) - Analysis of YAML deserialization vulnerabilities in TensorFlow.

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
- **Documentation**: [Bypassing picklescan: Sonatype discovers four vulnerabilities](https://www.sonatype.com/blog/bypassing-picklescan-sonatype-discovers-four-vulnerabilities)
- **Hugging Face Documentation**: [Pickle Scanning](https://huggingface.co/docs/hub/en/security-pickle)





## Supply Chain Attacks

### Academic Research

- [A Large-Scale Exploit Instrumentation Study of AI/ML Supply Chain Attacks in Hugging Face Models](https://arxiv.org/abs/2410.04490) - Comprehensive study of serialization vulnerabilities across Hugging Face by researchers from University of Notre Dame and University of Hawaii.
- [Supply-Chain Attacks in Machine Learning Frameworks](https://openreview.net/attachment?id=EH5PZW6aCr&name=pdf) - Analysis of ML supply chain security beyond algorithmic-level attacks.
- [Securing Machine Learning Supply Chains](https://apps.dtic.mil/sti/trecms/pdf/AD1164482.pdf) - Research on ML supply chain vulnerabilities and mitigations.
- [Towards Measuring Malicious Code Poisoning Attacks on Pre-trained Model Hubs](https://arxiv.org/html/2409.09368v1) - First systematic study of malicious code poisoning attacks on pre-trained model hubs.
- [Zero-Trust Artificial Intelligence Model Security](https://arxiv.org/html/2503.01758v1) - Research on securing AI models against supply chain attacks.

### Industry Reports

- [Malicious AI Models Undermine Software Supply-Chain Security](https://cacm.acm.org/research/malicious-ai-models-undermine-software-supply-chain-security/) - ACM report on emerging threats from malicious AI models.
- [AI Supply Chain Security: Hugging Face Malicious ML Models](https://nsfocusglobal.com/ai-supply-chain-security-hugging-face-malicious-ml-models/) - Analysis of malicious models on Hugging Face by NSFocus.
- [Hugging Face partners with Wiz Research to Improve AI Security](https://huggingface.co/blog/hugging-face-wiz-security-blog) - Collaborative efforts to improve AI model security.
- [The 2025 Software Supply Chain Security Report](https://ntsc.org/wp-content/uploads/2025/03/The-2025-Software-Supply-Chain-Security-Report-RL-compressed.pdf) - ReversingLabs' report on software supply chain security, including ML models.
- [Hugging Face Vulnerabilities Highlight AI-as-a-Service Risks](https://www.bankinfosecurity.com/hugging-face-vulnerabilities-highlight-ai-as-a-service-risks-a-24807) - Analysis of vulnerabilities in AI-as-a-Service platforms.

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

### Individual Researchers

#### Beatrice Casey (University of Notre Dame)

- **Focus**: ML supply chain security
- **Key Research**: [A Large-Scale Exploit Instrumentation Study of AI/ML Supply Chain Attacks in Hugging Face Models](https://arxiv.org/abs/2410.04490)


## Mitigation Strategies

- **Use Safer Formats**: Prefer formats like SafeTensors or ONNX over pickle-based formats when possible.
- **Implement Scanning**: Integrate model scanning into CI/CD pipelines and before loading any model.
- **Sandboxing**: Run model loading in isolated environments with limited permissions.
- **Verification**: Implement cryptographic verification of model origins.
- **Format Conversion**: Convert models to safer formats before deployment.
- **Code Review**: Review custom operators and Lambda layers for security issues.
- **Dependency Management**: Keep ML libraries and dependencies updated.
- **Input Validation**: Validate all inputs to ML models.
- **Principle of Least Privilege**: Run ML systems with minimal required permissions.
- **Monitoring**: Monitor ML systems for unusual behavior.


## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to suggest additions or improvements to this list.

## License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
