{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/mohammadilz12/Assignment1/blob/main/Assignment_02_Python_Packages_mohammadilz12_py.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 📚 Assignment 2 — NumPy, pandas & Matplotlib Essentials\n",
        "<p align=\"center\">📢⚠️📂  </p>\n",
        "\n",
        "<p align=\"center\"> Please name your file using the format: <code>assignmentName_nickname.py/.ipynb</code> (e.g., <code>project1_ali.py</code>) and push it to GitHub with a clear commit message.</p>\n",
        "\n",
        "<p align=\"center\"> 🚨📝🧠 </p>\n",
        "\n",
        "\n",
        "------------------------------------------------\n",
        "\n",
        "\n",
        "Welcome to your first data-science sprint! In this three-part mini-project you’ll touch the libraries every ML practitioner leans on daily:\n",
        "\n",
        "1. NumPy — fast n-dimensional arrays\n",
        "\n",
        "2. pandas — tabular data wrangling\n",
        "\n",
        "3. Matplotlib — quick, customizable plots\n",
        "\n",
        "Each part starts with “Quick-start notes”, then gives you two bite-sized tasks. Replace every # TODO with working Python in a notebook or script, run it, and check your results. Happy hacking! 😊\n"
      ],
      "metadata": {
        "id": "ZS5lje_18xC2"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 1 · NumPy 🧮\n",
        "Quick-start notes\n",
        "Core object: ndarray (n-dimensional array)\n",
        "\n",
        "Create data: np.array, np.arange, np.random.*\n",
        "\n",
        "Summaries: mean, std, sum, max, …\n",
        "\n",
        "Vectorised math beats Python loops for speed\n",
        "\n"
      ],
      "metadata": {
        "id": "u5vvtK-6840I"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Task 1 — Mock temperatures\n",
        "\n"
      ],
      "metadata": {
        "id": "UyNHtkGm9OgH"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lmaiboJe8E15",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b11b1cab-e564-4d04-b8bf-b8ae516a374d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[23.19467949 17.97347904 16.59395474 20.76983049 16.6958593  15.20725028\n",
            " 14.87062689 19.54876646 19.69205814 23.65657093]\n",
            "Std Dev: 5.001362672732942\n"
          ]
        }
      ],
      "source": [
        "# 👉 # TODO: import numpy and create an array of 365\n",
        "# normally-distributed °C values (µ=20, σ=5) called temps\n",
        "import numpy as np\n",
        "temps = np.random.normal(loc=20, scale=5, size=365)\n",
        "\n",
        "print(temps[:10])\n",
        "print(\"Std Dev:\", temps.std())\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Task 2 — Average temperature\n"
      ],
      "metadata": {
        "id": "BtsB8QKM9Xs_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 👉 # TODO: print the mean of temps\n",
        "print(\"Average:\", temps.mean())"
      ],
      "metadata": {
        "id": "DSR8aS-F9Z10",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c07771a5-e73c-45c2-d708-bc8c8b04e5dd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Average: 20.10576384942333\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 2 · pandas 📊\n",
        "Quick-start notes\n",
        "Main structures: DataFrame, Series\n",
        "\n",
        "Read data: pd.read_csv, pd.read_excel, …\n",
        "\n",
        "Selection: .loc[label], .iloc[pos]\n",
        "\n",
        "Group & summarise: .groupby(...).agg(...)\n",
        "\n"
      ],
      "metadata": {
        "id": "8JjHX4wy9dPz"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Task 3 — Load ride log\n"
      ],
      "metadata": {
        "id": "wRyJyhbt9uUr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 👉 # TODO: read \"rides.csv\" into df\n",
        "# (columns: date,temp,rides,weekday)\n",
        "import pandas as pd\n",
        "from google.colab import files\n",
        "uploaded = files.upload()\n",
        "df = pd.read_csv(\"rides.csv\")\n",
        "print(df.head())\n"
      ],
      "metadata": {
        "id": "1J4jcLct9yVO",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 183
        },
        "outputId": "f03a4fdd-4744-466a-cfa3-c5d0c3783c40"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-e2c076e7-ea97-45a5-a1b8-e40d564dd4f0\" name=\"files[]\" multiple disabled\n",
              "        style=\"border:none\" />\n",
              "     <output id=\"result-e2c076e7-ea97-45a5-a1b8-e40d564dd4f0\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script>// Copyright 2017 Google LLC\n",
              "//\n",
              "// Licensed under the Apache License, Version 2.0 (the \"License\");\n",
              "// you may not use this file except in compliance with the License.\n",
              "// You may obtain a copy of the License at\n",
              "//\n",
              "//      http://www.apache.org/licenses/LICENSE-2.0\n",
              "//\n",
              "// Unless required by applicable law or agreed to in writing, software\n",
              "// distributed under the License is distributed on an \"AS IS\" BASIS,\n",
              "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
              "// See the License for the specific language governing permissions and\n",
              "// limitations under the License.\n",
              "\n",
              "/**\n",
              " * @fileoverview Helpers for google.colab Python module.\n",
              " */\n",
              "(function(scope) {\n",
              "function span(text, styleAttributes = {}) {\n",
              "  const element = document.createElement('span');\n",
              "  element.textContent = text;\n",
              "  for (const key of Object.keys(styleAttributes)) {\n",
              "    element.style[key] = styleAttributes[key];\n",
              "  }\n",
              "  return element;\n",
              "}\n",
              "\n",
              "// Max number of bytes which will be uploaded at a time.\n",
              "const MAX_PAYLOAD_SIZE = 100 * 1024;\n",
              "\n",
              "function _uploadFiles(inputId, outputId) {\n",
              "  const steps = uploadFilesStep(inputId, outputId);\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  // Cache steps on the outputElement to make it available for the next call\n",
              "  // to uploadFilesContinue from Python.\n",
              "  outputElement.steps = steps;\n",
              "\n",
              "  return _uploadFilesContinue(outputId);\n",
              "}\n",
              "\n",
              "// This is roughly an async generator (not supported in the browser yet),\n",
              "// where there are multiple asynchronous steps and the Python side is going\n",
              "// to poll for completion of each step.\n",
              "// This uses a Promise to block the python side on completion of each step,\n",
              "// then passes the result of the previous step as the input to the next step.\n",
              "function _uploadFilesContinue(outputId) {\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  const steps = outputElement.steps;\n",
              "\n",
              "  const next = steps.next(outputElement.lastPromiseValue);\n",
              "  return Promise.resolve(next.value.promise).then((value) => {\n",
              "    // Cache the last promise value to make it available to the next\n",
              "    // step of the generator.\n",
              "    outputElement.lastPromiseValue = value;\n",
              "    return next.value.response;\n",
              "  });\n",
              "}\n",
              "\n",
              "/**\n",
              " * Generator function which is called between each async step of the upload\n",
              " * process.\n",
              " * @param {string} inputId Element ID of the input file picker element.\n",
              " * @param {string} outputId Element ID of the output display.\n",
              " * @return {!Iterable<!Object>} Iterable of next steps.\n",
              " */\n",
              "function* uploadFilesStep(inputId, outputId) {\n",
              "  const inputElement = document.getElementById(inputId);\n",
              "  inputElement.disabled = false;\n",
              "\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  outputElement.innerHTML = '';\n",
              "\n",
              "  const pickedPromise = new Promise((resolve) => {\n",
              "    inputElement.addEventListener('change', (e) => {\n",
              "      resolve(e.target.files);\n",
              "    });\n",
              "  });\n",
              "\n",
              "  const cancel = document.createElement('button');\n",
              "  inputElement.parentElement.appendChild(cancel);\n",
              "  cancel.textContent = 'Cancel upload';\n",
              "  const cancelPromise = new Promise((resolve) => {\n",
              "    cancel.onclick = () => {\n",
              "      resolve(null);\n",
              "    };\n",
              "  });\n",
              "\n",
              "  // Wait for the user to pick the files.\n",
              "  const files = yield {\n",
              "    promise: Promise.race([pickedPromise, cancelPromise]),\n",
              "    response: {\n",
              "      action: 'starting',\n",
              "    }\n",
              "  };\n",
              "\n",
              "  cancel.remove();\n",
              "\n",
              "  // Disable the input element since further picks are not allowed.\n",
              "  inputElement.disabled = true;\n",
              "\n",
              "  if (!files) {\n",
              "    return {\n",
              "      response: {\n",
              "        action: 'complete',\n",
              "      }\n",
              "    };\n",
              "  }\n",
              "\n",
              "  for (const file of files) {\n",
              "    const li = document.createElement('li');\n",
              "    li.append(span(file.name, {fontWeight: 'bold'}));\n",
              "    li.append(span(\n",
              "        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +\n",
              "        `last modified: ${\n",
              "            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :\n",
              "                                    'n/a'} - `));\n",
              "    const percent = span('0% done');\n",
              "    li.appendChild(percent);\n",
              "\n",
              "    outputElement.appendChild(li);\n",
              "\n",
              "    const fileDataPromise = new Promise((resolve) => {\n",
              "      const reader = new FileReader();\n",
              "      reader.onload = (e) => {\n",
              "        resolve(e.target.result);\n",
              "      };\n",
              "      reader.readAsArrayBuffer(file);\n",
              "    });\n",
              "    // Wait for the data to be ready.\n",
              "    let fileData = yield {\n",
              "      promise: fileDataPromise,\n",
              "      response: {\n",
              "        action: 'continue',\n",
              "      }\n",
              "    };\n",
              "\n",
              "    // Use a chunked sending to avoid message size limits. See b/62115660.\n",
              "    let position = 0;\n",
              "    do {\n",
              "      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);\n",
              "      const chunk = new Uint8Array(fileData, position, length);\n",
              "      position += length;\n",
              "\n",
              "      const base64 = btoa(String.fromCharCode.apply(null, chunk));\n",
              "      yield {\n",
              "        response: {\n",
              "          action: 'append',\n",
              "          file: file.name,\n",
              "          data: base64,\n",
              "        },\n",
              "      };\n",
              "\n",
              "      let percentDone = fileData.byteLength === 0 ?\n",
              "          100 :\n",
              "          Math.round((position / fileData.byteLength) * 100);\n",
              "      percent.textContent = `${percentDone}% done`;\n",
              "\n",
              "    } while (position < fileData.byteLength);\n",
              "  }\n",
              "\n",
              "  // All done.\n",
              "  yield {\n",
              "    response: {\n",
              "      action: 'complete',\n",
              "    }\n",
              "  };\n",
              "}\n",
              "\n",
              "scope.google = scope.google || {};\n",
              "scope.google.colab = scope.google.colab || {};\n",
              "scope.google.colab._files = {\n",
              "  _uploadFiles,\n",
              "  _uploadFilesContinue,\n",
              "};\n",
              "})(self);\n",
              "</script> "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Saving rides.csv to rides (1).csv\n",
            "       date  temp  rides    weekday\n",
            "0  9/1/2025    28    120     Monday\n",
            "1  9/2/2025    30    135    Tuesday\n",
            "2  9/3/2025    27    110  Wednesday\n",
            "3  9/4/2025    25     95   Thursday\n",
            "4  9/5/2025    26    100     Friday\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Task 4 — Weekday averages"
      ],
      "metadata": {
        "id": "byKd8SFK9w2y"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "mean_rides_per_weekday = df.groupby(\"weekday\")[\"rides\"].mean()\n",
        "\n",
        "print(mean_rides_per_weekday)"
      ],
      "metadata": {
        "id": "4rLrxkPj90p3",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8ce52056-4b74-4d11-a16b-19dd8afca1a9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "weekday\n",
            "Friday       100.0\n",
            "Monday       120.0\n",
            "Saturday      80.0\n",
            "Sunday        70.0\n",
            "Thursday      95.0\n",
            "Tuesday      135.0\n",
            "Wednesday    110.0\n",
            "Name: rides, dtype: float64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 3 · Matplotlib 📈\n",
        "Quick-start notes\n",
        "Workhorse: pyplot interface (import matplotlib.pyplot as plt)\n",
        "\n",
        "Figure & axes: fig, ax = plt.subplots()\n",
        "\n",
        "Common plots: plot, scatter, hist, imshow\n",
        "\n",
        "Display inline in Jupyter with %matplotlib inline or %matplotlib notebook"
      ],
      "metadata": {
        "id": "gbGMbtLf94M4"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Task 5 — Scatter plot"
      ],
      "metadata": {
        "id": "QOsToPh2-AnZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 👉 # TODO: scatter-plot temperature (x) vs rides (y) from df\n",
        "\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "plt.figure(figsize=(8,5))\n",
        "plt.scatter(df['temp'], df['rides'], color='blue', alpha=0.7)\n",
        "plt.title(\"Scatter Plot: Temperature vs Rides\")\n",
        "plt.xlabel(\"Temperature (°C)\")\n",
        "plt.ylabel(\"Number of Rides\")\n",
        "plt.grid(True)\n"
      ],
      "metadata": {
        "id": "_pCU2mIH-DAi",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 487
        },
        "outputId": "4e80d981-f55e-4d90-e0fd-1f6a2bc5caa1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 800x500 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAArcAAAHWCAYAAABt3aEVAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAVhhJREFUeJzt3X98zfX///H7GcdsY7/8mjG/RvnRkJ9tEYVJveVXIXpHCRUJ737wfkfoXUqKsH69qyXvvJOSdz/eYaikLZEfyZTfVH6HjY057Pn943x3Ph3bOMfZdubV7Xq57LK9nq/Xeb0e5+E17l7neV7HZowxAgAAACwgwN8FAAAAAEWFcAsAAADLINwCAADAMgi3AAAAsAzCLQAAACyDcAsAAADLINwCAADAMgi3AAAAsAzCLQAAACyDcAvAMvbs2SObzaa3337b36UAPunYsaM6dux4ye2+/PJL2Ww2ffnll8VeE3ClINwCV4DNmzfr9ttvV+3atVW+fHnVqFFDXbp00ezZs4vtmPPnz9fMmTPzje/fv1+TJk3Sxo0bi+3YF8r7Bzzvy263q169err77ru1a9euIjlGamqqJk2apBMnTlzW4ydNmuRWY2FfngQWq0pPT9ekSZO0Z88ef5dS4t5++22386Bs2bKqUaOGBg8erN9++83f5QGWUtbfBQC4uNTUVN14442qVauWhg4dqqioKP3yyy/69ttv9dJLL+mhhx4qluPOnz9fP/74o0aPHu02vn//fk2ePFl16tRR8+bNi+XYhRk1apRat24th8Oh9evX6/XXX9dnn32mzZs3Kzo62qd9p6amavLkyRo8eLDCw8O9fnzv3r1Vv3591/KpU6f0wAMPqFevXurdu7drvFq1aj7VeSVLT0/X5MmT1bFjR9WpU8ff5fjFlClTVLduXZ05c0bffvut3n77ba1evVo//vijypcv79pu2bJlfqwSuLIRboFS7umnn1ZYWJjWrl2bL3QdPnzYP0UVg6ysLIWEhFx0m/bt2+v222+XJN1zzz266qqrNGrUKM2dO1fjx48viTIL1bRpUzVt2tS1fPToUT3wwANq2rSp7rrrLj9WVnw8+TP7M9XhiW7duqlVq1aSpPvuu0+VK1fWc889p48//lh9+/Z1bVeuXDl/lQhc8ZiWAJRyO3fuVJMmTQq8mli1atV8Y//+97/Vpk0bBQcHKyIiQjfccIPbVaD//ve/uvXWWxUdHa3AwEDFxsbqqaee0vnz513bdOzYUZ999pn27t3rehm1Tp06+vLLL9W6dWtJznCZt+6Pc1zXrFmjm2++WWFhYQoODlaHDh30zTffuNWY9xJ+enq6BgwYoIiICLVr187r3tx0002SpN27d190u5UrV6p9+/YKCQlReHi4evTooa1bt7rV8+ijj0qS6tat63peeS+fHz16VD/99JOys7O9rvFCP/30k26//XZFRkaqfPnyatWqlT7++GO3bfJewl69erVGjRqlKlWqKDw8XMOHD9fZs2d14sQJ3X333YqIiFBERIQee+wxGWNcj8+bezx9+nTNmDFDtWvXVlBQkDp06KAff/zRp5q++uorPfjgg6patapq1qwpSdq7d68efPBBXX311QoKClKlSpV0xx13uE0/ePvtt3XHHXdIkm688UZXj/PmitpsNk2aNClfbXXq1NHgwYM9qkOSPv/8c9efdcWKFXXrrbdqy5YtF/0zWbdunWw2m+bOnZtv3dKlS2Wz2fTpp59Kkk6ePKnRo0erTp06CgwMVNWqVdWlSxetX7/+oscoTPv27SU5f8//qKA5t7/++qt69uypkJAQVa1aVWPGjFFOTk6B+/Xk97ConwtQWnDlFijlateurbS0NP3444+65pprLrrt5MmTNWnSJCUkJGjKlCkqV66c1qxZo5UrVyoxMVGSMxxUqFBBY8eOVYUKFbRy5UpNnDhRmZmZev755yVJ//jHP5SRkaFff/1VM2bMkCRVqFBBjRo10pQpUzRx4kQNGzbM9Q9zQkKCJGeI7Natm1q2bKknn3xSAQEBSk5O1k033aSvv/5abdq0cav3jjvuUIMGDfTMM8+4hTNP5QWCSpUqFbrN8uXL1a1bN9WrV0+TJk3S6dOnNXv2bF1//fVav3696tSpo969e2vbtm36z3/+oxkzZqhy5cqSpCpVqkiS5syZo8mTJ+uLL77wac7sli1bdP3116tGjRoaN26cQkJC9P7776tnz5768MMP1atXL7ftH3roIUVFRWny5Mn69ttv9frrrys8PFypqamqVauWnnnmGf3vf//T888/r2uuuUZ333232+PfeecdnTx5UiNGjNCZM2f00ksv6aabbtLmzZtd0yO8renBBx9UlSpVNHHiRGVlZUmS1q5dq9TUVPXv3181a9bUnj179Morr6hjx45KT09XcHCwbrjhBo0aNUqzZs3S3//+dzVq1EiSXN+9VVAd8+bN06BBg9S1a1c999xzys7O1iuvvKJ27dppw4YNhU6FaNWqlerVq6f3339fgwYNclu3YMECRUREqGvXrpKk+++/Xx988IFGjhypxo0b6/fff9fq1au1detWtWjRwuvnkfcfgIiIiItud/r0aXXq1En79u3TqFGjFB0drXnz5mnlypX5tvX097ConwtQahgApdqyZctMmTJlTJkyZUx8fLx57LHHzNKlS83Zs2fdttu+fbsJCAgwvXr1MufPn3dbl5ub6/o5Ozs73zGGDx9ugoODzZkzZ1xjt956q6ldu3a+bdeuXWskmeTk5HzHaNCggenatWu+49WtW9d06dLFNfbkk08aSebOO+/0qAdffPGFkWTeeustc+TIEbN//37z2WefmTp16hibzWbWrl1rjDFm9+7d+Wpr3ry5qVq1qvn9999dY5s2bTIBAQHm7rvvdo09//zzRpLZvXt3vuPn1fvFF194VK8xxhw5csRIMk8++aRrrFOnTiYuLs6tz7m5uSYhIcE0aNDANZacnGwk5etlfHy8sdls5v7773eNnTt3ztSsWdN06NDBNZbXh6CgIPPrr7+6xtesWWMkmTFjxlx2Te3atTPnzp1ze64FnVNpaWlGknnnnXdcYwsXLiy0jxf2Kk/t2rXNoEGDLlnHyZMnTXh4uBk6dKjb4w8ePGjCwsLyjV9o/Pjxxm63m2PHjrnGcnJyTHh4uLn33ntdY2FhYWbEiBEX3VdB8upevny5OXLkiPnll1/MBx98YKpUqWICAwPNL7/84rZ9hw4d3P5MZ86caSSZ999/3zWWlZVl6tev79ZTb34PL/e5AKUd0xKAUq5Lly5KS0vTbbfdpk2bNmnatGnq2rWratSo4fbS8eLFi5Wbm6uJEycqIMD9V9tms7l+DgoKcv188uRJHT16VO3bt1d2drZ++umny65z48aN2r59uwYMGKDff/9dR48e1dGjR5WVlaVOnTpp1apVys3NdXvM/fff79Ux7r33XlWpUkXR0dG69dZblZWVpblz57rmMF7owIED2rhxowYPHqzIyEjXeNOmTdWlSxf973//8+i4kyZNkjHGp6u2x44d08qVK9W3b19X348eParff/9dXbt21fbt2/O9a37IkCFuf3Zt27aVMUZDhgxxjZUpU0atWrUq8K4RPXv2VI0aNVzLbdq0Udu2bV3P+3JqGjp0qMqUKeM29sdzyuFw6Pfff1f9+vUVHh5ebC9xX1hHSkqKTpw4oTvvvNP1PI4ePaoyZcqobdu2+uKLLy66v379+snhcGjRokWusWXLlunEiRPq16+fayw8PFxr1qzR/v37L6vuzp07q0qVKoqJidHtt9+ukJAQffzxx25TKwryv//9T9WrV3fNOZek4OBgDRs2zG07b34PfX0uQGnFtATgCtC6dWstWrRIZ8+e1aZNm/TRRx9pxowZuv3227Vx40Y1btxYO3fuVEBAgBo3bnzRfW3ZskVPPPGEVq5cqczMTLd1GRkZl13j9u3bJSnfy7oX7v+PL7/WrVvXq2NMnDhR7du3V5kyZVS5cmU1atRIZcsW/tfY3r17JUlXX311vnWNGjXS0qVLS+zNSDt27JAxRhMmTNCECRMK3Obw4cNuYbRWrVpu68PCwiRJMTEx+caPHz+eb38NGjTIN3bVVVfp/fffv+yaCvozO336tKZOnark5GT99ttvblNMfDmnLubCOvLOv7x52BcKDQ296P6aNWumhg0basGCBa7/PCxYsECVK1d22+e0adM0aNAgxcTEqGXLlrrlllt09913q169eh7VnZSUpKuuukoZGRl66623tGrVKgUGBl7ycXv37lX9+vXd/rMj5T+3vfk99PW5AKUV4Ra4gpQrV06tW7dW69atddVVV+mee+7RwoUL9eSTT3r0+BMnTqhDhw4KDQ3VlClTFBsbq/Lly2v9+vV6/PHH811Z9UbeY59//vlCbxFWoUIFt+U/XvHzRFxcnDp37nxZ9flbXn8eeeQR1/zNC/3xVmKS8l0hvdi4uYw5y5dTU0F/Zg899JCSk5M1evRoxcfHKywsTDabTf379/fpnJLk9kbHi9WRd5x58+YpKioq3/YX+09Qnn79+unpp5/W0aNHVbFiRX388ce688473R7bt29ftW/fXh999JGWLVum559/Xs8995wWLVqkbt26XfIYbdq0cb3S0LNnT7Vr104DBgzQzz//nO/343J483vo63MBSivCLXCFyvsH8sCBA5Kk2NhY5ebmKj09vdB/1L788kv9/vvvWrRokW644QbXeEF3G7jwCtGlxmNjYyU5r5CVlgBau3ZtSdLPP/+cb91PP/2kypUr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G3X06FH961//Ut++fbVmzZp8odYb48eP19ixY13LmZmZiomJUWJiokJDQ4ui7ItyOBxKSUlRly5dZLfbi/14VkP/fEcPfUcPfUP/fEcPfUP/fFfSPcx7pf1SSn24DQkJUf369VW/fn1dd911atCggd58802NHz9eUVFROnz4sNv2586d07FjxxQVFVXoPgMDAxUYGJhv3G63l+gJXtLHsxr65zt66Dt66Bv65zt66Bv657uS6qGnx7ji7nObm5vrmi8bHx+vEydO6Pvvv3etX7lypXJzc9W2bVt/lQgAAAA/8euV21OnTmnHjh2u5d27d2vjxo2KjIxUpUqV9PTTT+u2225T9erVdfToUSUlJem3337THXfcIUlq1KiRbr75Zg0dOlSvvvqqHA6HRo4cqf79+3OnBAAAgD8hv4bbdevW6cYbb3Qt582DHTRokF599VX99NNPmjt3ro4ePapKlSqpdevW+vrrr9WkSRPXY959912NHDlSnTp1UkBAgPr06aNZs2aV+HMBAACA//k13Hbs2FHGmELXL1q06JL7iIyM1Pz584uyLAAAAFyhrrg5twAAAEBhCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAABXmNxcKT3d+XN6unMZgJNfw+2qVavUvXt3RUdHy2azafHixa51DodDjz/+uOLi4hQSEqLo6Gjdfffd2r9/v9s+jh07poEDByo0NFTh4eEaMmSITp06VcLPBACAkpGaKt11lzR8uHN5+HDncmqqf+sCSgu/htusrCw1a9ZMSUlJ+dZlZ2dr/fr1mjBhgtavX69Fixbp559/1m233ea23cCBA7VlyxalpKTo008/1apVqzRs2LCSegoAAJSY1FTpkUek9eulsDDnWFiYtGGDc5yAC0hl/Xnwbt26qVu3bgWuCwsLU0pKitvYnDlz1KZNG+3bt0+1atXS1q1btWTJEq1du1atWrWSJM2ePVu33HKLpk+frujo6GJ/DgAAlITcXGnOHOnYMal+falcOed4SIgUGyvt3CklJUnXXScFMOkQf2J+DbfeysjIkM1mU3h4uCQpLS1N4eHhrmArSZ07d1ZAQIDWrFmjXr16FbifnJwc5eTkuJYzMzMlOadCOByO4nsC/1/eMUriWFZE/3xHD31HD31D/7yXni7t2iXVru0Mtna7s3d532vVcgbczZulxo39WemVgXPQdyXdQ0+Pc8WE2zNnzujxxx/XnXfeqdDQUEnSwYMHVbVqVbftypYtq8jISB08eLDQfU2dOlWTJ0/ON75s2TIFBwcXbeEXceGVaXiH/vmOHvqOHvqG/nln/Pj8YwMGuPdwzx7nFzzDOei7kuphdna2R9tdEeHW4XCob9++MsbolVde8Xl/48eP19ixY13LmZmZiomJUWJiois4FyeHw6GUlBR16dJFdru92I9nNfTPd/TQd/TQN/TPe+npzjePhYU5pyLY7Q4NGJCi+fO7yOGwKytLysiQXnuNK7ee4Bz0XUn3MO+V9ksp9eE2L9ju3btXK1eudAufUVFROnz4sNv2586d07FjxxQVFVXoPgMDAxUYGJhv3G63l+gJXtLHsxr65zt66Dt66Bv657m4OKlePeebx2Jj/2/c4bDr7Fm79u2TWrRwbsecW89xDvqupHro6TFK9emfF2y3b9+u5cuXq1KlSm7r4+PjdeLECX3//feusZUrVyo3N1dt27Yt6XIBACg2AQHSyJFSRIRzbm1WlnM8K8u5HBEhjRhBsAX8euX21KlT2rFjh2t59+7d2rhxoyIjI1W9enXdfvvtWr9+vT799FOdP3/eNY82MjJS5cqVU6NGjXTzzTdr6NChevXVV+VwODRy5Ej179+fOyUAACwnIUGaPt1514Rdu5xjGRnOK7YjRjjXA392fg2369at04033uhazpsHO2jQIE2aNEkff/yxJKl58+Zuj/viiy/UsWNHSdK7776rkSNHqlOnTgoICFCfPn00a9asEqkfAICSlpDgvN3X5s3ON4699hpTEYA/8mu47dixo4wxha6/2Lo8kZGRmj9/flGWBQBAqRYQ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Rx8Vui4tLU2zZs1Sbm5ukRQFAAAAXA6Pw22PHj3yjf38888aN26cPvnkEw0cOFBTpkwp0uIAAAAAb1zWnNv9+/dr6NChiouL07lz57Rx40bNnTtXtWvXLur6AAAAAI95FW4zMjL0+OOPq379+tqyZYtWrFihTz75RNdcc01x1QcAAAB4zONpCdOmTdNzzz2nqKgo/ec//ylwmgIAAADgTx6H23HjxikoKEj169fX3LlzNXfu3AK3W7RoUZEVBwAAAHjD43B79913X/JWYAAAAIA/eRxu33777WIsAwAAAPAdn1AGAAAAyyDcAgAAwDIItwAAALAMwi0AAAAso9SH25MnT2r06NGqXbu2goKClJCQoLVr17rWG2M0ceJEVa9eXUFBQercubO2b9/ux4oBAADgL6U+3N53331KSUnRvHnztHnzZiUmJqpz58767bffJDk/XGLWrFl69dVXtWbNGoWEhKhr1646c+aMnysHAABASSvV4fb06dP68MMPNW3aNN1www2qX7++Jk2apPr16+uVV16RMUYzZ87UE088oR49eqhp06Z65513tH//fi1evNjf5QMAAKCEeXyfW384d+6czp8/r/Lly7uNBwUFafXq1dq9e7cOHjyozp07u9aFhYWpbdu2SktLU//+/Qvcb05OjnJyclzLmZmZkiSHwyGHw1EMz8Rd3jFK4lhWRP98Rw99Rw99Q/98Rw99Q/98V9I99PQ4NmOMKeZafJKQkKBy5cpp/vz5qlatmv7zn/9o0KBBql+/vpKTk3X99ddr//79ql69uusxffv2lc1m04IFCwrc56RJkzR58uR84/Pnz1dwcHCxPRcAAABcnuzsbA0YMEAZGRkKDQ0tdLtSfeVWkubNm6d7771XNWrUUJkyZdSiRQvdeeed+v777y97n+PHj9fYsWNdy5mZmYqJiVFiYuJFm1VUHA6HUlJS1KVLF9nt9mI/ntXQP9/RQ9/RQ9/QP9/RQ9/QP9+VdA/zXmm/lFIfbmNjY/XVV18pKytLmZmZql69uvr166d69eopKipKknTo0CG3K7eHDh1S8+bNC91nYGCgAgMD843b7fYSPcFL+nhWQ/98Rw99Rw99Q/98Rw99Q/98V1I99PQYpfoNZX8UEhKi6tWr6/jx41q6dKl69OihunXrKioqSitWrHBtl5mZqTVr1ig+Pt6P1QIAAMAf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pnLlymbLli2E2z+rESNGmJo1a5pdu3ZdctucnBwTHBxs5s+fXwKVXTkK6+HDDz/s+sso70uSCQgIMB06dPBPsaWQN+fgqVOnjCSzZMmSEqjsylFYD3ft2mUkmXnz5rmN9+3b1wwYMKAkSyzVPDkH33nnHWO3283hw4dLsLIrR2E93LFjh5FkfvzxR7fxTp06meHDh5dkiaWaJ+fgkSNHzPHjx40xxlSrVs1MmzathKq78nTq1MkMGzbMdUEkr295atWqZV588UX/FGeYc1tsjDEaOXKkPvroI61cuVJ169b16DHGGOXk5JRAhaXfpXo4btw4/fDDD9q4caPrS5JmzJih5ORkP1RculzOOZjXw+rVqxdzdVeGS/WwTp06io6OzndroW3btql27dolWWqp5M05+Oabb+q2225TlSpVSrDC0u9SPcybI3/h3TnKlCnDu/3l3TlYuXJlhYeHa+XKlTp8+LBuu+22Eqz0ypKbm6ucnBy1bNlSdrtdK1ascK37+eeftW/fPsXHx/uvQL/Faot74IEHTFhYmPnyyy/NgQMHXF/Z2dnGGGN27txpnnnmGbNu3Tqzd+9e880335ju3bubyMhIc+jQIT9XXzpcqocFEdMSXC7Vvx07dpgpU6aYdevWmd27d5v//ve/pl69euaGG27wc+Wlhyfn4IwZM0xoaKhZuHCh2b59u3niiSdM+fLlmdphPP8d3r59u7HZbObzzz/3U6Wl16V6ePbsWVO/fn3Tvn17s2bNGrNjxw4zffp0Y7PZXPMi/8w8OQffeustk5aWZnbs2GHmzZtnIiMjzdixY/1Ydekybtw489VXX5ndu3ebH374wYwbN87YbDazbNkyY4xzuketWrXMypUrzbp160x8fLyJj4/3a82E22IiqcCv5ORkY4wxv/32m+nWrZupWrWqsdvtpmbNmmbAgAHmp59+8m/hpcileljYYwi3Tpfq3759+8wNN9xgIiMjTWBgoKlfv7559NFH3W6J82fn6Tk4depUU7NmTRMcHGzi4+PN119/7Z+CSxlP+zd+/HgTExNjzp8/759CSzFPerht2zbTu3dvU7VqVRMcHGyaNm2a79Zgf1ae9O/xxx831apVM3a73TRo0MC88MILJjc3139FlzL33nuvqV27tilXrpypUqWK6dSpkyvYGmPM6dOnzYMPPmgiIiJMcHCw6dWrlzlw4IAfKzbGZowxxXNNGAAAAChZzLkFAACAZRBuAQAAYBmEWwAAAFgG4RYAAACWQbgFAACAZRBuAQAAYBmEWwAAAFgG4RYAAACWQbgFABS7CRMmaNiwYUW2v7Nnz6pOnTpat25dke0TgDUQbgH8adhstot+TZo0yd8lFrk6depo5syZfq3h4MGDeumll/SPf/zDNZaVlaX+/furevXquvPOO5WdnZ3vMQ899JDq1aunwMBAxcTEqHv37lqxYoUkqVy5cnrkkUf0+OOPl+hzAVD6EW4B/GkcOHDA9TVz5kyFhoa6jT3yyCP+LtEjxhidO3euRI959uzZy37sG2+8oYSEBNWuXds1NnPmTFWoUEHLli1TUFCQWwDfs2ePWrZsqZUrV+r555/X5s2btWTJEt14440aMWKEa7uBAwdq9erV2rJly2XXBsB6CLcA/jSioqJcX2FhYbLZbG5j7733nho1aqTy5curYcOGevnll12P3bNnj2w2m95//321b99eQUFBat26tbZt26a1a9eqVatWqlChgrp166YjR464Hjd48GD17NlTkydPVpUqVRQaGqr777/fLSzm5uZq6tSpqlu3roKCgtSsWTN98MEHrvVffvmlbDabPv/8c7Vs2VKBgYFavXq1du7cqR49eqhatWqqUKGCWrdureXLl7se17FjR+3du1djxoxxXZ2WpEmTJql58+ZuvZk5c6bq1KmTr+6nn35a0dHRuvrqqyVJv/zyi/r27avw8HBFRkaqR48e2rNnz0X7/t5776l79+5uY8ePH9dVV12luLg4NWzYUCdOnHCte/DBB2Wz2fTdd9+pT58+uuqqq9SkSRONHTtW3377rWu7iIgIXX/99XrvvfcuenwAfy6EWwCQ9O6772rixIl6+umntXXrVj3zzDOaMGGC5s6d67bdk08+qSeeeELr169X2bJlNWDAAD322GN66aWX9PXXX2vHjh2aOHGi22NWrFihrVu36ssvv9R//vMfLVq0SJMnT3atnzp1qt555x29+uqr2rJli8aMGaO77rpLX331ldt+xo0bp2effVZbt25V06ZNderUKd1yyy1asWKFNmzYoJtvvlndu3fXvn37JEmLFi1SzZo1NWXKFNfVaW+sWLFCP//8s1JSUvTpp5/K4XCoa9euqlixor7++mt98803qlChgm6++eZCr+weO3ZM6enpatWqldv4yJEj9dprr8lutys5OVkPP/ywa/slS5ZoxIgRCgkJybe/8PBwt+U2bdro66+/9up5AbA4AwB/QsnJySYsLMy1HBsba+bPn++2zVNPPWXi4+ONMcbs3r3bSDJvvPGGa/1//vMfI8msWLHCNTZ16lRz9dVXu5YHDRpkIiMjTVZWlmvslVdeMRUqVDDnz583Z86cMcHBwSY1NdXt2EOGDDF33nmnMcaYL774wkgyixcvvuTzatKkiZk9e7ZruXbt2mbGjBlu2zz55JOmWbNmbmMzZswwtWvXdqu7WrVqJicnxzU2b948c/XVV5vc3FzXWE5OjgkKCjJLly4tsJ4NGzYYSWbfvn351p0/f94cOHDAbX9r1qwxksyiRYsu+VyNMeall14yderU8WhbAH8OZf2arAGgFMjKytLOnTs1ZMgQDR061DV+7tw5hYWFuW3btGlT18/VqlWTJMXFxbmNHT582O0xzZo1U3BwsGs5Pj5ep06d0i+//KJTp04pOztbXbp0cXvM2bNnde2117qNXXj189SpU5o0aZI+++wzHThwQOfOndPp06ddV259FRcXp3LlyrmWN23apB07dqhixYpu2505c0Y7d+4scB+nT5+WJJUvXz7fuoCAAEVFRbmNGWO8qjEoKCjfm9EA/LkRbgH86Z06dUqS9K9//Utt27Z1W1emTBm3Zbvd7vo5bw7rhWO5ubleH/uzzz5TjRo13NYFBga6LV/4Mv0jjzyilJQUTZ8+XfXr11dQUJBuv/32S775KyAgIF+IdDgc+ba78HinTp1Sy5Yt9e677+bbtkqVKgUeq3LlypKcc2wL2+aPGjRoIJvNpp9++umS20rOaQye7BfAnwfhFsCfXrVq1RQdHa1du3Zp4MCBRb7/TZs26fTp0woKCpIkffvtt6pQoYJiYmIUGRmpwMBA7du3Tx06dPBqv998840GDx6sXr16SXKGzwvf3FWuXDmdP3/ebaxKlSo6ePCgjDGugL5x48ZLHq9FixZasGCBqlatqtDQUI9qjI2NVWhoqNLT03XVVVddcvvIyEh17dpVSUlJGjVqVL6AfeLECbd5tz/++GO+K9wA/tx4QxkASJo8ebKmTp2qWbNmadu2bdq8ebOSk5P14osv+rzvs2fPasiQIUpPT9f//vc/Pfnkkxo5cqQCAgJUsWJFPfLIIxozZozmzp2rnTt3av369Zo9e3a+N7NdqEGDBlq0aJE2btyoTZs2acCAAfmuGtepU0erVq3Sb7/9pqNHj0py3kXhyJEjmjZtmnbu3KmkpCR9/vnnl3weAwcOVOXKldWjRw99/fXX2r17t7788kuNGjVKv/76a4GPCQgIUOfOnbV69WoPuyUlJSXp/PnzatOmjT788ENt375dW7du1axZsxQfH++27ddff63ExESP9w3A+gi3ACDpvvvu0xtvvKHk5GTFxcWpQ4cOevvtt1W3bl2f992pUyc1aNBAN9xwg/r166fbbrvN7QMjnnrqKU2YMEFTp05Vo0aNdPPNN+uzzz675LFffPFFRUREKCEhQd27d1fXrl3VokULt22mTJmiPXv2KDY21vXyfaNGjfTyyy8rKSlJzZo103fffefRPX6Dg4O1atUq1apVS71791ajRo00ZMgQnTlz5qJXcu+77z699957Hk/XqFevntavX68bb7xRf/vb33TNNdeoS5cuWrFihV555RXXdmlpacrIyNDtt9/u0X4B/DnYjLez9wEAHhs8eLBOnDihxYsX+7sUvzHGqG3bthozZozuvPPOIttvv3791KxZM/39738vsn0CuPJx5RYAUKxsNptef/31Iv1UtbNnzyouLk5jxowpsn0CsAau3AJAMeLKLQCULMItAAAALINpCQAAALAMwi0AAAAsg3ALAAAAyyDcAgAAwDIItwAAALAMwi0AAAAsg3ALAAAAyyDcAgAAwDL+H0179muPvQJ4AAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Task 6 — Show the figure\n",
        "\n"
      ],
      "metadata": {
        "id": "TBGXIzVV-E5u"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 👉 # TODO: call plt.show() so the plot appears\n",
        "\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "plt.figure(figsize=(8,5))\n",
        "plt.scatter(df['temp'], df['rides'], color='blue', alpha=0.7)\n",
        "plt.title(\"Scatter Plot: Temperature vs Rides\")\n",
        "plt.xlabel(\"Temperature (°C)\")\n",
        "plt.ylabel(\"Number of Rides\")\n",
        "plt.grid(True)\n",
        "\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "ycvsNyqh-GRM",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 487
        },
        "outputId": "aa327a7b-55a7-4125-f590-2a16a118f51b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 800x500 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAArcAAAHWCAYAAABt3aEVAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAVhhJREFUeJzt3X98zfX///H7GcdsY7/8mjG/RvnRkJ9tEYVJveVXIXpHCRUJ737wfkfoXUqKsH69qyXvvJOSdz/eYaikLZEfyZTfVH6HjY057Pn943x3Ph3bOMfZdubV7Xq57LK9nq/Xeb0e5+E17l7neV7HZowxAgAAACwgwN8FAAAAAEWFcAsAAADLINwCAADAMgi3AAAAsAzCLQAAACyDcAsAAADLINwCAADAMgi3AAAAsAzCLQAAACyDcAvAMvbs2SObzaa3337b36UAPunYsaM6dux4ye2+/PJL2Ww2ffnll8VeE3ClINwCV4DNmzfr9ttvV+3atVW+fHnVqFFDXbp00ezZs4vtmPPnz9fMmTPzje/fv1+TJk3Sxo0bi+3YF8r7Bzzvy263q169err77ru1a9euIjlGamqqJk2apBMnTlzW4ydNmuRWY2FfngQWq0pPT9ekSZO0Z88ef5dS4t5++22386Bs2bKqUaOGBg8erN9++83f5QGWUtbfBQC4uNTUVN14442qVauWhg4dqqioKP3yyy/69ttv9dJLL+mhhx4qluPOnz9fP/74o0aPHu02vn//fk2ePFl16tRR8+bNi+XYhRk1apRat24th8Oh9evX6/XXX9dnn32mzZs3Kzo62qd9p6amavLkyRo8eLDCw8O9fnzv3r1Vv3591/KpU6f0wAMPqFevXurdu7drvFq1aj7VeSVLT0/X5MmT1bFjR9WpU8ff5fjFlClTVLduXZ05c0bffvut3n77ba1evVo//vijypcv79pu2bJlfqwSuLIRboFS7umnn1ZYWJjWrl2bL3QdPnzYP0UVg6ysLIWEhFx0m/bt2+v222+XJN1zzz266qqrNGrUKM2dO1fjx48viTIL1bRpUzVt2tS1fPToUT3wwANq2rSp7rrrLj9WVnw8+TP7M9XhiW7duqlVq1aSpPvuu0+VK1fWc889p48//lh9+/Z1bVeuXDl/lQhc8ZiWAJRyO3fuVJMmTQq8mli1atV8Y//+97/Vpk0bBQcHKyIiQjfccIPbVaD//ve/uvXWWxUdHa3AwEDFxsbqqaee0vnz513bdOzYUZ999pn27t3rehm1Tp06+vLLL9W6dWtJznCZt+6Pc1zXrFmjm2++WWFhYQoODlaHDh30zTffuNWY9xJ+enq6BgwYoIiICLVr187r3tx0002SpN27d190u5UrV6p9+/YKCQlReHi4evTooa1bt7rV8+ijj0qS6tat63peeS+fHz16VD/99JOys7O9rvFCP/30k26//XZFRkaqfPnyatWqlT7++GO3bfJewl69erVGjRqlKlWqKDw8XMOHD9fZs2d14sQJ3X333YqIiFBERIQee+wxGWNcj8+bezx9+nTNmDFDtWvXVlBQkDp06KAff/zRp5q++uorPfjgg6patapq1qwpSdq7d68efPBBXX311QoKClKlSpV0xx13uE0/ePvtt3XHHXdIkm688UZXj/PmitpsNk2aNClfbXXq1NHgwYM9qkOSPv/8c9efdcWKFXXrrbdqy5YtF/0zWbdunWw2m+bOnZtv3dKlS2Wz2fTpp59Kkk6ePKnRo0erTp06CgwMVNWqVdWlSxetX7/+oscoTPv27SU5f8//qKA5t7/++qt69uypkJAQVa1aVWPGjFFOTk6B+/Xk97ConwtQWnDlFijlateurbS0NP3444+65pprLrrt5MmTNWnSJCUkJGjKlCkqV66c1qxZo5UrVyoxMVGSMxxUqFBBY8eOVYUKFbRy5UpNnDhRmZmZev755yVJ//jHP5SRkaFff/1VM2bMkCRVqFBBjRo10pQpUzRx4kQNGzbM9Q9zQkKCJGeI7Natm1q2bKknn3xSAQEBSk5O1k033aSvv/5abdq0cav3jjvuUIMGDfTMM8+4hTNP5QWCSpUqFbrN8uXL1a1bN9WrV0+TJk3S6dOnNXv2bF1//fVav3696tSpo969e2vbtm36z3/+oxkzZqhy5cqSpCpVqkiS5syZo8mTJ+uLL77wac7sli1bdP3116tGjRoaN26cQkJC9P7776tnz5768MMP1atXL7ftH3roIUVFRWny5Mn69ttv9frrrys8PFypqamqVauWnnnmGf3vf//T888/r2uuuUZ333232+PfeecdnTx5UiNGjNCZM2f00ksv6aabbtLmzZtd0yO8renBBx9UlSpVNHHiRGVlZUmS1q5dq9TUVPXv3181a9bUnj179Morr6hjx45KT09XcHCwbrjhBo0aNUqzZs3S3//+dzVq1EiSXN+9VVAd8+bN06BBg9S1a1c999xzys7O1iuvvKJ27dppw4YNhU6FaNWqlerVq6f3339fgwYNclu3YMECRUREqGvXrpKk+++/Xx988IFGjhypxo0b6/fff9fq1au1detWtWjRwuvnkfcfgIiIiItud/r0aXXq1En79u3TqFGjFB0drXnz5mnlypX5tvX097ConwtQahgApdqyZctMmTJlTJkyZUx8fLx57LHHzNKlS83Zs2fdttu+fbsJCAgwvXr1MufPn3dbl5ub6/o5Ozs73zGGDx9ugoODzZkzZ1xjt956q6ldu3a+bdeuXWskmeTk5HzHaNCggenatWu+49WtW9d06dLFNfbkk08aSebOO+/0qAdffPGFkWTeeustc+TIEbN//37z2WefmTp16hibzWbWrl1rjDFm9+7d+Wpr3ry5qVq1qvn9999dY5s2bTIBAQHm7rvvdo09//zzRpLZvXt3vuPn1fvFF194VK8xxhw5csRIMk8++aRrrFOnTiYuLs6tz7m5uSYhIcE0aNDANZacnGwk5etlfHy8sdls5v7773eNnTt3ztSsWdN06NDBNZbXh6CgIPPrr7+6xtesWWMkmTFjxlx2Te3atTPnzp1ze64FnVNpaWlGknnnnXdcYwsXLiy0jxf2Kk/t2rXNoEGDLlnHyZMnTXh4uBk6dKjb4w8ePGjCwsLyjV9o/Pjxxm63m2PHjrnGcnJyTHh4uLn33ntdY2FhYWbEiBEX3VdB8upevny5OXLkiPnll1/MBx98YKpUqWICAwPNL7/84rZ9hw4d3P5MZ86caSSZ999/3zWWlZVl6tev79ZTb34PL/e5AKUd0xKAUq5Lly5KS0vTbbfdpk2bNmnatGnq2rWratSo4fbS8eLFi5Wbm6uJEycqIMD9V9tms7l+DgoKcv188uRJHT16VO3bt1d2drZ++umny65z48aN2r59uwYMGKDff/9dR48e1dGjR5WVlaVOnTpp1apVys3NdXvM/fff79Ux7r33XlWpUkXR0dG69dZblZWVpblz57rmMF7owIED2rhxowYPHqzIyEjXeNOmTdWlSxf973//8+i4kyZNkjHGp6u2x44d08qVK9W3b19X348eParff/9dXbt21fbt2/O9a37IkCFuf3Zt27aVMUZDhgxxjZUpU0atWrUq8K4RPXv2VI0aNVzLbdq0Udu2bV3P+3JqGjp0qMqUKeM29sdzyuFw6Pfff1f9+vUVHh5ebC9xX1hHSkqKTpw4oTvvvNP1PI4ePaoyZcqobdu2+uKLLy66v379+snhcGjRokWusWXLlunEiRPq16+fayw8PFxr1qzR/v37L6vuzp07q0qVKoqJidHtt9+ukJAQffzxx25TKwryv//9T9WrV3fNOZek4OBgDRs2zG07b34PfX0uQGnFtATgCtC6dWstWrRIZ8+e1aZNm/TRRx9pxowZuv3227Vx40Y1btxYO3fuVEBAgBo3bnzRfW3ZskVPPPGEVq5cqczMTLd1GRkZl13j9u3bJSnfy7oX7v+PL7/WrVvXq2NMnDhR7du3V5kyZVS5cmU1atRIZcsW/tfY3r17JUlXX311vnWNGjXS0qVLS+zNSDt27JAxRhMmTNCECRMK3Obw4cNuYbRWrVpu68PCwiRJMTEx+caPHz+eb38NGjTIN3bVVVfp/fffv+yaCvozO336tKZOnark5GT99ttvblNMfDmnLubCOvLOv7x52BcKDQ296P6aNWumhg0basGCBa7/PCxYsECVK1d22+e0adM0aNAgxcTEqGXLlrrlllt09913q169eh7VnZSUpKuuukoZGRl66623tGrVKgUGBl7ycXv37lX9+vXd/rMj5T+3vfk99PW5AKUV4Ra4gpQrV06tW7dW69atddVVV+mee+7RwoUL9eSTT3r0+BMnTqhDhw4KDQ3VlClTFBsbq/Lly2v9+vV6/PHH811Z9UbeY59//vlCbxFWoUIFt+U/XvHzRFxcnDp37nxZ9flbXn8eeeQR1/zNC/3xVmKS8l0hvdi4uYw5y5dTU0F/Zg899JCSk5M1evRoxcfHKywsTDabTf379/fpnJLk9kbHi9WRd5x58+YpKioq3/YX+09Qnn79+unpp5/W0aNHVbFiRX388ce688473R7bt29ftW/fXh999JGWLVum559/Xs8995wWLVqkbt26XfIYbdq0cb3S0LNnT7Vr104DBgzQzz//nO/343J483vo63MBSivCLXCFyvsH8sCBA5Kk2NhY5ebmKj09vdB/1L788kv9/vvvWrRokW644QbXeEF3G7jwCtGlxmNjYyU5r5CVlgBau3ZtSdLPP/+cb91PP/2kypUr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G3X06FH961//Ut++fbVmzZp8odYb48eP19ixY13LmZmZiomJUWJiokJDQ4ui7ItyOBxKSUlRly5dZLfbi/14VkP/fEcPfUcPfUP/fEcPfUP/fFfSPcx7pf1SSn24DQkJUf369VW/fn1dd911atCggd58802NHz9eUVFROnz4sNv2586d07FjxxQVFVXoPgMDAxUYGJhv3G63l+gJXtLHsxr65zt66Dt66Bv65zt66Bv657uS6qGnx7ji7nObm5vrmi8bHx+vEydO6Pvvv3etX7lypXJzc9W2bVt/lQgAAAA/8euV21OnTmnHjh2u5d27d2vjxo2KjIxUpUqV9PTTT+u2225T9erVdfToUSUlJem3337THXfcIUlq1KiRbr75Zg0dOlSvvvqqHA6HRo4cqf79+3OnBAAAgD8hv4bbdevW6cYbb3Qt582DHTRokF599VX99NNPmjt3ro4ePapKlSqpdevW+vrrr9WkSRPXY959912NHDlSnTp1UkBAgPr06aNZs2aV+HMBAACA//k13Hbs2FHGmELXL1q06JL7iIyM1Pz584uyLAAAAFyhrrg5twAAAEBhCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAABXmNxcKT3d+XN6unMZgJNfw+2qVavUvXt3RUdHy2azafHixa51DodDjz/+uOLi4hQSEqLo6Gjdfffd2r9/v9s+jh07poEDByo0NFTh4eEaMmSITp06VcLPBACAkpGaKt11lzR8uHN5+HDncmqqf+sCSgu/htusrCw1a9ZMSUlJ+dZlZ2dr/fr1mjBhgtavX69Fixbp559/1m233ea23cCBA7VlyxalpKTo008/1apVqzRs2LCSegoAAJSY1FTpkUek9eulsDDnWFiYtGGDc5yAC0hl/Xnwbt26qVu3bgWuCwsLU0pKitvYnDlz1KZNG+3bt0+1atXS1q1btWTJEq1du1atWrWSJM2ePVu33HKLpk+frujo6GJ/DgAAlITcXGnOHOnYMal+falcOed4SIgUGyvt3CklJUnXXScFMOkQf2J+DbfeysjIkM1mU3h4uCQpLS1N4eHhrmArSZ07d1ZAQIDWrFmjXr16FbifnJwc5eTkuJYzMzMlOadCOByO4nsC/1/eMUriWFZE/3xHD31HD31D/7yXni7t2iXVru0Mtna7s3d532vVcgbczZulxo39WemVgXPQdyXdQ0+Pc8WE2zNnzujxxx/XnXfeqdDQUEnSwYMHVbVqVbftypYtq8jISB08eLDQfU2dOlWTJ0/ON75s2TIFBwcXbeEXceGVaXiH/vmOHvqOHvqG/nln/Pj8YwMGuPdwzx7nFzzDOei7kuphdna2R9tdEeHW4XCob9++MsbolVde8Xl/48eP19ixY13LmZmZiomJUWJiois4FyeHw6GUlBR16dJFdru92I9nNfTPd/TQd/TQN/TPe+npzjePhYU5pyLY7Q4NGJCi+fO7yOGwKytLysiQXnuNK7ee4Bz0XUn3MO+V9ksp9eE2L9ju3btXK1eudAufUVFROnz4sNv2586d07FjxxQVFVXoPgMDAxUYGJhv3G63l+gJXtLHsxr65zt66Dt66Bv657m4OKlePeebx2Jj/2/c4bDr7Fm79u2TWrRwbsecW89xDvqupHro6TFK9emfF2y3b9+u5cuXq1KlSm7r4+PjdeLECX3//feusZUrVyo3N1dt27Yt6XIBACg2AQHSyJFSRIRzbm1WlnM8K8u5HBEhjRhBsAX8euX21KlT2rFjh2t59+7d2rhxoyIjI1W9enXdfvvtWr9+vT799FOdP3/eNY82MjJS5cqVU6NGjXTzzTdr6NChevXVV+VwODRy5Ej179+fOyUAACwnIUGaPt1514Rdu5xjGRnOK7YjRjjXA392fg2369at04033uhazpsHO2jQIE2aNEkff/yxJKl58+Zuj/viiy/UsWNHSdK7776rkSNHqlOnTgoICFCfPn00a9asEqkfAICSlpDgvN3X5s3ON4699hpTEYA/8mu47dixo4wxha6/2Lo8kZGRmj9/flGWBQBAqRYQ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Rx8Vui4tLU2zZs1Sbm5ukRQFAAAAXA6Pw22PHj3yjf38888aN26cPvnkEw0cOFBTpkwp0uIAAAAAb1zWnNv9+/dr6NChiouL07lz57Rx40bNnTtXtWvXLur6AAAAAI95FW4zMjL0+OOPq379+tqyZYtWrFihTz75RNdcc01x1QcAAAB4zONpCdOmTdNzzz2nqKgo/ec//ylwmgIAAADgTx6H23HjxikoKEj169fX3LlzNXfu3AK3W7RoUZEVBwAAAHjD43B79913X/JWYAAAAIA/eRxu33777WIsAwAAAPAdn1AGAAAAyyDcAgAAwDIItwAAALAMwi0AAAAso9SH25MnT2r06NGqXbu2goKClJCQoLVr17rWG2M0ceJEVa9eXUFBQercubO2b9/ux4oBAADgL6U+3N53331KSUnRvHnztHnzZiUmJqpz58767bffJDk/XGLWrFl69dVXtWbNGoWEhKhr1646c+aMnysHAABASSvV4fb06dP68MMPNW3aNN1www2qX7++Jk2apPr16+uVV16RMUYzZ87UE088oR49eqhp06Z65513tH//fi1evNjf5QMAAKCEeXyfW384d+6czp8/r/Lly7uNBwUFafXq1dq9e7cOHjyozp07u9aFhYWpbdu2SktLU//+/Qvcb05OjnJyclzLmZmZkiSHwyGHw1EMz8Rd3jFK4lhWRP98Rw99Rw99Q/98Rw99Q/98V9I99PQ4NmOMKeZafJKQkKBy5cpp/vz5qlatmv7zn/9o0KBBql+/vpKTk3X99ddr//79ql69uusxffv2lc1m04IFCwrc56RJkzR58uR84/Pnz1dwcHCxPRcAAABcnuzsbA0YMEAZGRkKDQ0tdLtSfeVWkubNm6d7771XNWrUUJkyZdSiRQvdeeed+v777y97n+PHj9fYsWNdy5mZmYqJiVFiYuJFm1VUHA6HUlJS1KVLF9nt9mI/ntXQP9/RQ9/RQ9/QP9/RQ9/QP9+VdA/zXmm/lFIfbmNjY/XVV18pKytLmZmZql69uvr166d69eopKipKknTo0CG3K7eHDh1S8+bNC91nYGCgAgMD843b7fYSPcFL+nhWQ/98Rw99Rw99Q/98Rw99Q/98V1I99PQYpfoNZX8UEhKi6tWr6/jx41q6dKl69OihunXrKioqSitWrHBtl5mZqTVr1ig+Pt6P1QIAAMAf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pnLlymbLli2E2z+rESNGmJo1a5pdu3ZdctucnBwTHBxs5s+fXwKVXTkK6+HDDz/s+sso70uSCQgIMB06dPBPsaWQN+fgqVOnjCSzZMmSEqjsylFYD3ft2mUkmXnz5rmN9+3b1wwYMKAkSyzVPDkH33nnHWO3283hw4dLsLIrR2E93LFjh5FkfvzxR7fxTp06meHDh5dkiaWaJ+fgkSNHzPHjx40xxlSrVs1MmzathKq78nTq1MkMGzbMdUEkr295atWqZV588UX/FGeYc1tsjDEaOXKkPvroI61cuVJ169b16DHGGOXk5JRAhaXfpXo4btw4/fDDD9q4caPrS5JmzJih5ORkP1RculzOOZjXw+rVqxdzdVeGS/WwTp06io6OzndroW3btql27dolWWqp5M05+Oabb+q2225TlSpVSrDC0u9SPcybI3/h3TnKlCnDu/3l3TlYuXJlhYeHa+XKlTp8+LBuu+22Eqz0ypKbm6ucnBy1bNlSdrtdK1ascK37+eeftW/fPsXHx/uvQL/Faot74IEHTFhYmPnyyy/NgQMHXF/Z2dnGGGN27txpnnnmGbNu3Tqzd+9e880335ju3bubyMhIc+jQIT9XXzpcqocFEdMSXC7Vvx07dpgpU6aYdevWmd27d5v//ve/pl69euaGG27wc+Wlhyfn4IwZM0xoaKhZuHCh2b59u3niiSdM+fLlmdphPP8d3r59u7HZbObzzz/3U6Wl16V6ePbsWVO/fn3Tvn17s2bNGrNjxw4zffp0Y7PZXPMi/8w8OQffeustk5aWZnbs2GHmzZtnIiMjzdixY/1Ydekybtw489VXX5ndu3ebH374wYwbN87YbDazbNkyY4xzuketWrXMypUrzbp160x8fLyJj4/3a82E22IiqcCv5ORkY4wxv/32m+nWrZupWrWqsdvtpmbNmmbAgAHmp59+8m/hpcileljYYwi3Tpfq3759+8wNN9xgIiMjTWBgoKlfv7559NFH3W6J82fn6Tk4depUU7NmTRMcHGzi4+PN119/7Z+CSxlP+zd+/HgTExNjzp8/759CSzFPerht2zbTu3dvU7VqVRMcHGyaNm2a79Zgf1ae9O/xxx831apVM3a73TRo0MC88MILJjc3139FlzL33nuvqV27tilXrpypUqWK6dSpkyvYGmPM6dOnzYMPPmgiIiJMcHCw6dWrlzlw4IAfKzbGZowxxXNNGAAAAChZzLkFAACAZRBuAQAAYBmEWwAAAFgG4RYAAACWQbgFAACAZRBuAQAAYBmEWwAAAFgG4RYAAACWQbgFABS7CRMmaNiwYUW2v7Nnz6pOnTpat25dke0TgDUQbgH8adhstot+TZo0yd8lFrk6depo5syZfq3h4MGDeumll/SPf/zDNZaVlaX+/furevXquvPOO5WdnZ3vMQ899JDq1aunwMBAxcTEqHv37lqxYoUkqVy5cnrkkUf0+OOPl+hzAVD6EW4B/GkcOHDA9TVz5kyFhoa6jT3yyCP+LtEjxhidO3euRI959uzZy37sG2+8oYSEBNWuXds1NnPmTFWoUEHLli1TUFCQWwDfs2ePWrZsqZUrV+r555/X5s2btWTJEt14440aMWKEa7uBAwdq9erV2rJly2XXBsB6CLcA/jSioqJcX2FhYbLZbG5j7733nho1aqTy5curYcOGevnll12P3bNnj2w2m95//321b99eQUFBat26tbZt26a1a9eqVatWqlChgrp166YjR464Hjd48GD17NlTkydPVpUqVRQaGqr777/fLSzm5uZq6tSpqlu3roKCgtSsWTN98MEHrvVffvmlbDabPv/8c7Vs2VKBgYFavXq1du7cqR49eqhatWqqUKGCWrdureXLl7se17FjR+3du1djxoxxXZ2WpEmTJql58+ZuvZk5c6bq1KmTr+6nn35a0dHRuvrqqyVJv/zyi/r27avw8HBFRkaqR48e2rNnz0X7/t5776l79+5uY8ePH9dVV12luLg4NWzYUCdOnHCte/DBB2Wz2fTdd9+pT58+uuqqq9SkSRONHTtW3377rWu7iIgIXX/99XrvvfcuenwAfy6EWwCQ9O6772rixIl6+umntXXrVj3zzDOaMGGC5s6d67bdk08+qSeeeELr169X2bJlNWDAAD322GN66aWX9PXXX2vHjh2aOHGi22NWrFihrVu36ssvv9R//vMfLVq0SJMnT3atnzp1qt555x29+uqr2rJli8aMGaO77rpLX331ldt+xo0bp2effVZbt25V06ZNderUKd1yyy1asWKFNmzYoJtvvlndu3fXvn37JEmLFi1SzZo1NWXKFNfVaW+sWLFCP//8s1JSUvTpp5/K4XCoa9euqlixor7++mt98803qlChgm6++eZCr+weO3ZM6enpatWqldv4yJEj9dprr8lutys5OVkPP/ywa/slS5ZoxIgRCgkJybe/8PBwt+U2bdro66+/9up5AbA4AwB/QsnJySYsLMy1HBsba+bPn++2zVNPPWXi4+ONMcbs3r3bSDJvvPGGa/1//vMfI8msWLHCNTZ16lRz9dVXu5YHDRpkIiMjTVZWlmvslVdeMRUqVDDnz583Z86cMcHBwSY1NdXt2EOGDDF33nmnMcaYL774wkgyixcvvuTzatKkiZk9e7ZruXbt2mbGjBlu2zz55JOmWbNmbmMzZswwtWvXdqu7WrVqJicnxzU2b948c/XVV5vc3FzXWE5OjgkKCjJLly4tsJ4NGzYYSWbfvn351p0/f94cOHDAbX9r1qwxksyiRYsu+VyNMeall14yderU8WhbAH8OZf2arAGgFMjKytLOnTs1ZMgQDR061DV+7tw5hYWFuW3btGlT18/VqlWTJMXFxbmNHT582O0xzZo1U3BwsGs5Pj5ep06d0i+//KJTp04pOztbXbp0cXvM2bNnde2117qNXXj189SpU5o0aZI+++wzHThwQOfOndPp06ddV259FRcXp3LlyrmWN23apB07dqhixYpu2505c0Y7d+4scB+nT5+WJJUvXz7fuoCAAEVFRbmNGWO8qjEoKCjfm9EA/LkRbgH86Z06dUqS9K9//Utt27Z1W1emTBm3Zbvd7vo5bw7rhWO5ubleH/uzzz5TjRo13NYFBga6LV/4Mv0jjzyilJQUTZ8+XfXr11dQUJBuv/32S775KyAgIF+IdDgc+ba78HinTp1Sy5Yt9e677+bbtkqVKgUeq3LlypKcc2wL2+aPGjRoIJvNpp9++umS20rOaQye7BfAnwfhFsCfXrVq1RQdHa1du3Zp4MCBRb7/TZs26fTp0woKCpIkffvtt6pQoYJiYmIUGRmpwMBA7du3Tx06dPBqv998840GDx6sXr16SXKGzwvf3FWuXDmdP3/ebaxKlSo6ePCgjDGugL5x48ZLHq9FixZasGCBqlatqtDQUI9qjI2NVWhoqNLT03XVVVddcvvIyEh17dpVSUlJGjVqVL6AfeLECbd5tz/++GO+K9wA/tx4QxkASJo8ebKmTp2qWbNmadu2bdq8ebOSk5P14osv+rzvs2fPasiQIUpPT9f//vc/Pfnkkxo5cqQCAgJUsWJFPfLIIxozZozmzp2rnTt3av369Zo9e3a+N7NdqEGDBlq0aJE2btyoTZs2acCAAfmuGtepU0erVq3Sb7/9pqNHj0py3kXhyJEjmjZtmnbu3KmkpCR9/vnnl3weAwcOVOXKldWjRw99/fXX2r17t7788kuNGjVKv/76a4GPCQgIUOfOnbV69WoPuyUlJSXp/PnzatOmjT788ENt375dW7du1axZsxQfH++27ddff63ExESP9w3A+gi3ACDpvvvu0xtvvKHk5GTFxcWpQ4cOevvtt1W3bl2f992pUyc1aNBAN9xwg/r166fbbrvN7QMjnnrqKU2YMEFTp05Vo0aNdPPNN+uzzz675LFffPFFRUREKCEhQd27d1fXrl3VokULt22mTJmiPXv2KDY21vXyfaNGjfTyyy8rKSlJzZo103fffefRPX6Dg4O1atUq1apVS71791ajRo00ZMgQnTlz5qJXcu+77z699957Hk/XqFevntavX68bb7xRf/vb33TNNdeoS5cuWrFihV555RXXdmlpacrIyNDtt9/u0X4B/DnYjLez9wEAHhs8eLBOnDihxYsX+7sUvzHGqG3bthozZozuvPPOIttvv3791KxZM/39738vsn0CuPJx5RYAUKxsNptef/31Iv1UtbNnzyouLk5jxowpsn0CsAau3AJAMeLKLQCULMItAAAALINpCQAAALAMwi0AAAAsg3ALAAAAyyDcAgAAwDIItwAAALAMwi0AAAAsg3ALAAAAyyDcAgAAwDL+H0179muPvQJ4AAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "code = \"\"\"\n",
        "# اینجا کد Python Assignment 02 قرار می‌گیرد\n",
        "print(\"Hello, World!\")\n",
        "\"\"\"\n",
        "\n",
        "with open(\"Assignment_02_Python_Packages_mohammadilz12.py\", \"w\") as f:\n",
        "    f.write(code)"
      ],
      "metadata": {
        "id": "D2uOJz93sbi3"
      },
      "execution_count": 2,
      "outputs": []
    }
  ]
}