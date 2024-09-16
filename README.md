## Show, Don't Tell: Aligning Language Models with Demonstrated Feedback

This repository contains source code for the paper **Show, Don't Tell: Aligning Language Models with Demonstrated Feedback** by [Omar Shaikh](https://oshaikh.com/), [Michelle Lam](https://michelle123lam.github.io/), [Joey Hejna](http://joeyhejna.com/), [Yijia Shao](https://cs.stanford.edu/~shaoyj/), [Michael Bernstein](https://hci.stanford.edu/msb/), and [Diyi Yang](https://cs.stanford.edu/~diyiy/). Feel free to reach out to [Omar Shaikh](https://oshaikh.com/) with any questions!

[[Paper]](https://arxiv.org/abs/2406.00888)

### *Abstract* 

Language models are aligned to emulate the collective voice of many, resulting in outputs that align with no one in particular. Steering LLMs away from generic output is possible through supervised finetuning or RLHF, but requires prohibitively large datasets for new ad-hoc tasks. We argue that it is instead possible to align an LLM to a specific setting by leveraging a very small number ($<10$) of demonstrations as feedback. Our method, Demonstration ITerated Task Optimization (DITTO), directly aligns language model outputs to a user's demonstrated behaviors. Derived using ideas from online imitation learning, DITTO cheaply generates online comparison data by treating users' demonstrations as preferred over output from the LLM and its intermediate checkpoints. We evaluate DITTO's ability to learn fine-grained style and task alignment across domains such as news articles, emails, and blog posts. Additionally, we conduct a user study soliciting a range of demonstrations from participants ($N=16$). Across our benchmarks and user study, we find that win-rates for DITTO outperform few-shot prompting, supervised fine-tuning, and other self-play methods by an average of 19\% points. By using demonstrations as feedback directly, DITTO offers a novel method for effective customization of LLMs.

### *Instructions*

We build on [alignment-handbook repo](https://github.com/huggingface/alignment-handbook). Here are the steps to get set up!

First, create a Python virtual environment using e.g. Conda:
```shell
conda create -n ditto python=3.10 && conda activate ditto
```

Next, install PyTorch `v2.1.2`. We used the following.

```shell
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then, install the alignment handbook dependencies.

```shell
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout 606d2e954fd17999af40e6fb4f712055ca11b2f0
python -m pip install .
```

Lastly, install the requirements for this repo to avoid errors due to updates in packages. 

```shell 
pip install -r requirements.txt
```

A sample shell script with training + generation is in run.sh (trains Mistral Instruct v0.2 7B). Right now, it's set to finetune on email examples. The shell script has an argument for trying different datasets in the paper. Note that you may need to change the config files for your specific hardware or dataset.

```shell 
bash run.sh
```

### Debugging

* `AttributeError: 'DittoConfig' object has no attribute 'packing'`: revert to older version of trl (`trl==0.8.6`) in `requirements.txt`. 


### *How do I cite this work?* 

Feel free to use the following BibTeX entry.

**BibTeX:**

```tex
@misc{shaikh2024show,
      title={Show, Don't Tell: Aligning Language Models with Demonstrated Feedback}, 
      author={Omar Shaikh and Michelle Lam and Joey Hejna and Yijia Shao and Michael Bernstein and Diyi Yang},
      year={2024},
      eprint={2406.00888},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

