        var categorySelect = document.getElementById("categorySelect");
        var resultDiv = document.getElementById("resultDiv");
        var categoryContent = {
            "Health and beauty": "Health and Beauty content goes here.",
            "Electronic accessories": "Electronic Accessories content goes here.",
            "Home and lifestyle": "Home and Lifestyle content goes here.",
            "Fashion accessories":"Fashion accessories content goes here.",
            "Food and beverages":"Food and beverages content goes here.",
            "Sports and travel" :"Sports and travel content goes here."
        };
        categorySelect.addEventListener("change", function() {
            var selectedCategory = categorySelect.value;
            resultDiv.innerHTML = "<p>" + categoryContent[selectedCategory] + "</p>";
        });

        const A1 = document.getElementById("A1");
        const image = document.getElementById("image");
        const image1 = document.getElementById("image1");
        const image2 = document.getElementById("image2");
        const image3 = document.getElementById("image3");
        A1.addEventListener("click", function() {
            image.style.display = "block";
            image1.style.display = "block";
            image2.style.display = "block";
            image3.style.display = "block";
        });