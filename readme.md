<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO
<br />
<div align="center">
  <a href="https://github.com/solutionfinders/SentenceExplainer">
    <img src="images/logo.png" alt="Logo" width="80" height="80"> 
  </a> -->

<h3 align="center">A Sentence Similarity Explainer</h3>

  <p align="center">
    An explainable AI module to explain the results of sentence similarity with Sentence Transformers
    <br />
    <!--<a href="https://github.com/solutionfinders/SentenceExplainer"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/solutionfinders/SentenceExplainer">View Demo</a> -->
    ·
    <a href="https://github.com/solutionfinders/SentenceExplainer/issues">Report Bug</a>
    ·
    <a href="https://github.com/solutionfinders/SentenceExplainer/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <!-- <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>-->
    </li> 
    <li><a href="#idea">Idea</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://lavrio.solutions)

An explainable AI module for sentence similarity.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

You need Sentence Transformers from huggingface, a model, and a corpus of texts/sentences. This package helps you to understand which words contribute to a Sentence Similarity score.

Standard output: the result of your search/sentence similarity, and under each hit the original query/sentence with the words in colors corresponding to the importance of the word for this hit. 

Red words are important for this hit (i.e. they reduce the score of this hit when removed from the query) while blue words are detrimental to this hit (when they are not part aóf the query, the score of this hit increases).


<!-- IDEA -->
## Idea

The idea is to first query the sentence similarity scores for a sentence in regard to a corpus of texts. The resulting list of hits is then analyzed by this script in the following way:

Take the first word out of the sentence. The resulting shortened query sentence is again embedded by the model, and used as a query. This time, we only look at the list of results from our first query, and calculate the differences in scores.

These differences are attributed to this first word. Subsequently, we are taking other words out of the query and compare the change in scores, until we finished it with the whole sentence.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Initial release
- [ ] Better Documentation
- [ ] Tests and packaging

See the [open issues](https://github.com/solutionfinders/SentenceExplainer/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



LICENSE
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Jens Beyer - [@codingGarden@mstdn.social](https://mstdn.social/@codingGarden) - jens@lavrio.solutions - [lavrio.solutions](https://lavrio.solutions)

Project Link: [https://github.com/solutionfinders/SentenceExplainer](https://github.com/solutionfinders/SentenceExplainer)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/solutionfinders/SentenceExplainer.svg?style=for-the-badge
[contributors-url]: https://github.com/solutionfinders/SentenceExplainer/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/solutionfinders/SentenceExplainer.svg?style=for-the-badge
[forks-url]: https://github.com/solutionfinders/SentenceExplainer/network/members
[stars-shield]: https://img.shields.io/github/stars/solutionfinders/SentenceExplainer.svg?style=for-the-badge
[stars-url]: https://github.com/solutionfinders/SentenceExplainer/stargazers
[issues-shield]: https://img.shields.io/github/issues/solutionfinders/SentenceExplainer.svg?style=for-the-badge
[issues-url]: https://github.com/solutionfinders/SentenceExplainer/issues
[license-shield]: https://img.shields.io/github/license/solutionfinders/SentenceExplainer.svg?style=for-the-badge
[license-url]: https://github.com/solutionfinders/SentenceExplainer/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/jens-beyer-94a209124
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 