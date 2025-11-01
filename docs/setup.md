# Getting started

## Introduction

A flexible benchmarking toolkit for streaming Spatio-Temporal Point-Process models. BenchSTPP is a modular, research-grade framework for end-to-end development, training, and evaluation of Spatio-Temporal Point-Process (STPP) models. It couples declarative YAML configuration with PyTorch Lightning execution, Ray Tune hyper-parameter optimisation, and version-controlled logging to deliver rapid prototyping and rigorous, reproducible benchmarking on streaming event data.

For more information visit Our [BenchSTPP](https://github.com/YahyaAalaila/STPPGC.git) repository on Github.

## Installation

### with git

BenchSTPP can be directly used from GitHub by cloning the repository into a subfolder of your project root which might be useful if you want to use the very latest version:

<h4>1. Clone the repository locally</h4>

Start by cloning the repository and navigating into its directory:

```bash
$ git clone git@github.com:YahyaAalaila/STPPGC.git
$ cd STPPGC

```

<h4>2. Create a virtual environment (optional but recommended)</h4>

It is highly recommended to use a virtual environment to prevent potential dependency conflicts with other Python projects.

```bash
# Create the Environment (named .venv)
$ python3 -m venv .venv

# Activate the Environment
# On WSL, Linux, & macOS
$ source .venv/bin/activate
```

!!! warning "For Windows users"
    It is recommended to use WSL 2 (Windows Subsystem for Linux) for a smoother experience, as some dependencies and commands may not work properly on native Windows.

<h4>3. Install dependencies</h4>
```bash
$ pip install -e .
# To enable the optional Neural-STPP models
$ pip install -e .[neural]
```

<h4>4. verify Installation</h4>

To verify that BenchSTPP is installed correctly, run a simple test with full YAML file using the following command:

```bash
$ pip install -e .
# To enable the optional Neural-STPP models
$ pip install -e .[neural]
```

This should start a training process using the Neuralstpp model without any errors.

### with pip

!!! info 
    this part is under construction

___

<div class="feedback-widget">
  <p>Was this content helpful?</p>
  <button class="feedback-btn like" aria-label="Like">yes </button>
  <span> / </span>
  <button class="feedback-btn dislike" aria-label="Dislike">no</button>
</div>

<script>
document.querySelector('.feedback-widget').addEventListener('click', function(event) {
  if (event.target.classList.contains('feedback-btn')) {
    const feedback = event.target.classList.contains('like') ? 'like' : 'dislike';
    console.log('User feedback:', feedback);
    const msg = document.createElement('span');
    msg.textContent = ' Thank you for your feedback!';
    msg.style.marginLeft = '1em';
    msg.style.fontWeight = 'bold';
    event.target.parentNode.appendChild(msg);
    setTimeout(() => msg.remove(), 3000);
  }
});
</script>
