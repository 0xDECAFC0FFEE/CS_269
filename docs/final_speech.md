# final speech

We're Lucas and Shelly and we're presenting our work on pruning meta learning algorithms

## Motivation

Some limitations on the deployment of machine learning systems on edge devices are as follows: Firstly, edge devices are frequently memory constrained. Large models are unable to be trained in these environments if they don't fit in the device's memory. Secondly, training models locally requires large amounts of energy and compute time. This could be detrimental to training models on battery powered edge devices such as phones or rural devices with intermittent power. Thirdly, some edge devices collect massive amounts of sensor data but are unable to upload it to cloud services due to privacy or cellular bandwidth constraints. 

## Meta Learning

In order to handle these challenges unique to edge computing, a possible solution comes in the form of meta learning. Meta learning is a branch of machine learning concerned with training meta models to finetune to new tasks quickly by training on other related tasks first. These approaches are well suited to the challenges common amongst edge devices as they allow powerful servers to train meta models that generalize to new tasks quickly. These models could then be deployed to a large fleet of small edge devices to fit to the local tasks. 

Amongst the many meta learning algorithms, we are currently focused on an approach known as Model Agnostic Meta Learning or MAML. It's model agnostic and designed to be fast to finetune. 

## LTH

We intend to couple MAML with the Lottery Ticket Hypothesis. It proposes a pruning algorithm that consistently uncovers sub-networks of any fully connected network with the same or greater test accuracy while retaining only a small fraction of the weights. These subnetworks are discovered through an iterative train - prune - reinitialization process. They empirically show that networks generally contain a higher scoring sparse subnetwork and that later iterations of this training process converge in very few epochs. 

## LTH

Pruning networks with this process generally leads to a very specific graph. The solid lines show that pruning the network results in higher accuracies initially as small weights are removed but as it becomes too sparse, the scores rapidly drop.

## Proposed Method
Applying this sparsification algorithm to MAML has several benefits to the earlier challenges common to ml in edge devices. The sparsity induced through the pruning algorithm could result in improvements to meta model finetuning time and accuracy. Not only are there fewer operations to do but models sparsified through the process have also been shown to converge in far fewer epochs than the original dense network. Also, the lottery ticket pruning process is entirely model agnostic. This synergizes well with MAML as both processes work for all networks with loss functions. 

## Related work
Prior work showed that applying mixup to maml would result in accuracy gains. We finished implementing Mixup but haven't finished tuning the hyperparameters in time for today.
Our baseline is a paper released in 2019 that sparsified maml with l1 regularization. In our experiments, we were able to achieve a 6% increase in testing acc at the same sparsity.
rigl?


## Results 1

Our current results are very promising. It's kind of hard to see with all the lines but pruning maml has the same rise then fall pattern mentioned earlier when explaining the lottery ticket hypothesis. Prune iterations 1 to 6 generally increase network scores but as weights are further pruned, the scores drop.

## Results 2

We were able to finetune the model on a raspberry pi on the full testing set. This graph from late december plots the prune iterations against the epoch with the maximum test accuracy. Each finetuning epoch took roughly 7 minutes. Theoretically, if we were to implement early stopping, the meta model would spend up to 5 times less time finetuning as it could trigger early stopping 2 epochs in. We noticed a few months ago that we were finetuning our model against the full testing set instead of just one task. Due to how the test set repeatedly finetunes on a 100 different tasks, we should be able to run our model on even cheaper computers, such as a $2 raspberry pi 0.

## Results 3

The results here are a bit noisy as we only finished one run but this graph shows the maximum finetuning test accuracy of each intermediate step of pruning the model. According to this graph, it doesn't make sense to sparsify our model on past 41% - the scores drop rapidly from there. Unfortunately, this doesn't result in a smaller memory footprint due to the overhead in sparse matrices. Of course, we could forcibly increase the sparsity by adding more weights but, without more work, leveraging the sparsity to decrease memory overhead doesn't make sense. After pruning the model, the accuracy rose by 1.53%.

## further work
While our scores increased with pruning, we intend to run more tests in the future. 
Specifically, one, we hope to smooth out the noise in our experiments with more runs. Our current results bounce all over the place. This could be due to 10% pruning being too high. Similarly, we hope to further improve our hyperparameters such that our unpruned model is on par with the original maml. Over the last few weeks, we improved its acc by 3% but we're still 1% short of the original maml's score. We had a few runs 2 days ago where the unpruned model's accuracy was 48.2% but stopped those in favor of trying to get metamix running. 

Secondly, we noticed that the training and validation scores are lower than the testing scores which suggests further inspection. 

Again, We didn't finish tuning the metamix's hyperparameters in time for the presentation but it's currently running in the background. Each full lth maml run takes a week to complete and metamix might take up to 4 times longer. 

Fourth, MAML’s accuracy has been superceded by MAML++ at 50% and MetaSGD at 51%. Implementing the lth for these algorithms could be useful as well and might result in state of the art accuracy. 

Fifth, we hope to run the finetuning suite on a raspberry pi 0. 

We also intend to train on 5 say 5 shot mini imagenet or other datasets but havent due to time constraints. It should take around 30 days at our current training rate to train the model.

And lastly, we could also look into speeding up the meta model training time. Currently, it takes 6 days to complete a full experiment but some of our earlier runs indicated that using rigl should decrease it to just half a day without much of a decrease in acc. 

## thanks for listening

things I fixed:
Our batchnorm running mean and std weren't updating correctly on the validation and testing support sets. (state_dict vs parameters)
added image augmentations 
    3 day model training times 
    execute 15 pruning iterations 
    > 40 days for an experiment
    now 6 days at ~8 hours/iter
    increased training speed significantly with hardware upgrades - cpu and ram clock speed bottlenecked
Added dataset shuffling
MAML requires second order gradient and we modeified the implementation to compute a first order approximation
significant increase in model complexity and various changes to the model architecture and other hyperparameters
Hyperparameters weren't tuned in previous run

