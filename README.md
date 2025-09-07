## Improving Sickle Cell Disease Classification: A Fusion of Conventional Classifiers, Segmented Images, and Convolutional Neural Networks

### Paper available <a href="https://sol.sbc.org.br/index.php/eniac/article/view/25713" target="_blank">here</a>

## Citation

```
@inproceedings{Cardoso2023,
 author = {Victor Cardoso and Rodrigo Moreira and João Mari and Larissa Rodrigues Moreira},
title = {Improving Sickle Cell Disease Classification: A Fusion of Conventional Classifiers, Segmented Images, and Convolutional Neural Networks},
 booktitle = {Anais do XX Encontro Nacional de Inteligência Artificial e Computacional},
 location = {Belo Horizonte/MG},
 year = {2023},
 keywords = {},
 issn = {2763-9061},
 pages = {345--358},
 publisher = {SBC},
 address = {Porto Alegre, RS, Brasil},
 doi = {10.5753/eniac.2023.234076},
 url = {https://sol.sbc.org.br/index.php/eniac/article/view/25713}
}


```

<meta name="citation_title" content="Improving Sickle Cell Disease Classification: A Fusion of Conventional Classifiers, Segmented Images, and Convolutional Neural Network" />
<meta name="citation_publication_date" content="2023" />
<meta name="citation_author" content="Victor Cardoso and Rodrigo Moreira and João Mari and Larissa Rodrigues Moreira" />

## Abstract
Sickle cell anemia, which is characterized by abnormal erythrocyte morphology, can be detected using microscopic images. Computational techniques in medicine enhance the diagnosis and treatment efficiency. However, many computational techniques, particularly those based on Convolutional Neural Networks (CNNs), require high resources and time for training, highlighting the research opportunities in methods with low computational overhead. In this paper, we propose a novel approach combining conventional classifiers, segmented images, and CNNs for the automated classification of sickle cell disease. We evaluated the impact of segmented images on classification, providing insight into deep learning integration. Our results demonstrate that using segmented images and CNN features with an SVM achieves an accuracy of 96.80\%. This finding is relevant for computationally efficient scenarios, paving the way for future research and advancements in medical-image analysis.

## 🚀 Web Application

This project includes a Streamlit web application for real-time erythrocyte classification. The app allows users to upload microscopic images of red blood cells or test with sample images from the dataset.

### Features
- **Interactive Demo**: Test with sample images from each class
- **File Upload**: Upload your own erythrocyte images
- **Real-time Classification**: Instant results with confidence scores
- **Educational Content**: Learn about sickle cell disease
- **Medical Disclaimer**: Appropriate safety warnings

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Streamlit Cloud Deployment
1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Set the main file path to `app.py`
6. Click Deploy!

**Note**: The app requires the model files (`svm_model.pkl`) and sample dataset to be included in the repository for full functionality.

![Steps of the proposed approach](steps.png)

## Acknowledgment
- FAPEMIG (Grant number CEX - APQ-02964-17)
- Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Finance Code 001.
- Federal University of Viçosa (UFV), Brazil

## Authors
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/victoralcantara75" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/27792114?v=4" width="100px;" alt="Victor J. Alcântara Cardoso"/><br>
        <sub>
          <b>Victor J. Alcântara Cardoso</b>
        </sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/larissafrodrigues" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/12631107?v=4" width="100px;" alt="Larissa F. Rodrigues Moreira"/><br>
        <sub>
          <b>Larissa F. Rodrigues Moreira</b>
        </sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/romoreira" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/15040641?v=4" width="100px;" alt="Rodrigo Moreira"/><br>
        <sub>
          <b>Rodrigo Moreira</b>
        </sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/joaofmari" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/23125037?v=4" width="100px;" alt="João Fernando Mari"/><br>
        <sub>
          <b>João Fernando Mari</b>
        </sub>
      </a>
    </td>   
  </tr>
</table>
