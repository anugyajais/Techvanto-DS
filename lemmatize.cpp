// #include <string>
#include <bits/stdc++.h>
// #include <unordered_map>

using namespace std;

string lemmatize(const string& word) {
	static const unordered_map<string, string> lemmaDict = {
		{"running", "run"},
		{"ran", "run"},
		{"cars", "car"},
		{"better", "good"},
		{"children", "child"},
		// Add more word mappings as needed
	};

	auto it = lemmaDict.find(word);
	if (it != lemmaDict.end()) {
		return it->second;
	}
	// Basic rule: remove 's' for plurals (very naive)
	if (word.size() > 2 && word.back() == 's') {
		return word.substr(0, word.size() - 1);
	}
	return word;
}

vector<string> apply_lemmatization(vector<string>& words){
	for(string& word : words) {
		transform(word.begin(), word.end(), word.begin(), ::tolower);
		word = lemmatize(word);
	}
}

int main() {
	vector<string> words = {"Running", "Cars", "Children", "Better", "Dogs"};
	apply_lemmatization(words);

	for(const auto& word : words) {
		cout << word << " ";
	}
	cout << endl;

	// Bag of Words function
	auto bag_of_words = [](const vector<string>& lemmatized_words) {
		unordered_map<string, int> bow;
		for(const auto& word : lemmatized_words) {
			bow[word]++;
		}
		return bow;
	};

	auto bow = bag_of_words(words);
	cout << "Bag of Words:" << endl;
	for(const auto& pair : bow) {
		cout << pair.first << ": " << pair.second << endl;
	}

	return 0;
}