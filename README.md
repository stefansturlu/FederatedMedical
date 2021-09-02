# Robust Federated Learning and Adversary Detection using Knowledge Distillation

This project constitutes a simulation environment for FL tasks that facilitates evaluating the performance of the different robust and privacy preserving FL techniques against different attack scenarios.
At the current stage, the project tackles several aspects of FL in adversarial settings.

- **Robust Aggregation FL**: currently there are 11 main aggregation schemes implemented: Federated Averaging (not robust), MKRUM, COMED, Adaptive Federated Averaging, FedMGDA+, Clustering, FedPADRC, FedDF, FedBE, FedRAD and FedABE.

- **Privacy preserving FL**: we have experimented with client level differential privacy and the syntactic approch for FL, that makes use of Information loss to perform generalisation over the training and testing datasets.

- **FL Adversial strategies**: We have simulated 3 different strategies attackers can have: label-flipping attacks, faulty attacks that send noisy parameter updates and free-riding attacks.

## Work built upon

*Samuel Trew - MEng Computing - Computation in Medicine and Biology: Enhancing Robust Aggregation in Federated Learning (2021)*

*Grama, M., Musat, M., Muñoz-González, L., Passerat-Palmbach, J., Rueckert, D., & Alansary, A. (2020). Robust Aggregation for Adaptive Privacy Preserving Federated Learning in Healthcare. arXiv preprint arXiv:2009.08294.*

## Relevant Paper

*Stefán Páll Sturluson (2021). Robust Federated Averaging and Adversary Detection using Knowledge Distillation

---

## Initial set-up

We recomend using Python 3.8.x for compatibility with all the used modules and libraries.
Simply running `python3 main.py` will run whichever experiment is chosen.
setupVenvOnGPUCluster.sh is the script that can be used for the initial setup of the python virtual enviroment; we have used it for the cloud-based virtual machines and the GPU cluster provided by from Imperial College Cloud

### Using the running scripts

runExperiment.sh and stopLastExperiment.sh can be used for running the script locally or on a virtual machine send write the output to file.
runExperimentGPU.sh can be used for running the experiment using the Slurm GPU Cluster from Imperial College Cloud

## Formatting

The formatting is done through Python Black, run `python3 -m black --line-length 100 .` to appropriately format the entire directory.

## Extending the project

To experiment with new datasets and models there are a few steps that need to be followed.

- Firstly, a new DatasetLoader child class needs to be created for the newly considered dataset. The single public method of this class, representing its only communication bridge to the other modules classes and modules is getDatasets method. This method, given a list of percentages representing the data split among clients, will return a set of datasets corresponding to the clients' data share, and a test dataset, which is later passed to the aggregator.

- In order to use a new model, this could be added to the classifiers directory within a dedicated module file. Finally, all of those elements could be put together within the main.py experiment setup methods, similarly to the already existing ones.

- We also provided a straight forward mechanism of extending the code base to use new aggregation schemes: creating child classes of the Aggregator class in Aggregator.py module. Simply add a new file in the aggregators folder.

- Any further configuration details can be added to either the federated system config in DefaultExperimentConfiguration.py or to the AggregatorConfig.py if it is aggregator specific configuration.

- When running the experiments, to change the configuration, use the CustomConfig.py file and call `scenario_conversion` like `for attackName in scenario_conversion():`

## Work Built Upon References

Within this project we have integrated concepts and implementations coming from several pieces of work related to the subject.

- *Luis Muñoz-González, Kenneth T Co, and Emil C Lupu. 2019. Byzantine-robust federated machine learning through adaptive model averaging. arXiv preprint arXiv:1909.05125 (2019)*

- *Peva Blanchard, Rachid Guerraoui, Julien Stainer, et al. 2017. Machine learning with adversaries: Byzantine tolerant gradient descent. In Advances in Neural Information Processing Systems. 119–129*

- *Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. 2018. Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates. In International Conference on Machine Learning. 5650–5659.*

- *Olivia Choudhury, Aris Gkoulalas-Divanis, Theodoros Salonidis, Issa Sylla, Yoonyoung Park, Grace Hsu, and Amar Das. 2020. Anonymizing Data for PrivacyPreserving Federated Learning. arXiv preprint arXiv:2002.09096 (2020).*

- *Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jun 2016. doi: 10.1109/cvpr.2016.90. Available: http://dx.doi.org/10.1109/CVPR.2016.90.*

- *Wenqi Li, Fausto Milletarì, Daguang Xu, Nicola Rieke, Jonny Hancox, Wentao Zhu, Maximilian Baust, Yan Cheng, Sébastien Ourselin, M Jorge Cardoso, et al. 2019. Privacy-preserving federated brain tumour segmentation. In International Workshop on Machine Learning in Medical Imaging. Springer, 133–141.*

- *Lindawangg/COVID-Net. Github, 2020. https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md*

## License

[MIT](https://choosealicense.com/licenses/mit/)
