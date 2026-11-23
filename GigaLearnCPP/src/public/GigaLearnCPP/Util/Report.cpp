#pragma once

#include "Report.h"
#include <algorithm>

void GGL::Report::Display(std::vector<std::string> keyRows) const {
	std::stringstream stream;
	stream << std::string(8, '\n');
	stream << RG_DIVIDER << std::endl;
	for (std::string row : keyRows) {
		if (!row.empty()) {

			int indentLevel = 0;
			while (row[0] == '-') {
				indentLevel++;
				row.erase(row.begin());
			}

			std::string prefix = {};
			if (indentLevel > 0) {
				prefix += std::string((indentLevel - 1) * 3, ' ');
				prefix += " - ";
			}
			if (Has(row)) {
				stream << prefix << SingleToString(row, true) << std::endl;
			} else {
				continue;
			}
		} else {
			stream << std::endl;
		}
	}

	// 2026-11-19: also print every metric NOT in the fixed list (sorted), so newly added rewards/skill metrics
	// (flip resets, flicks, dribbles...) show up in the console log instead of only in wandb.
	std::vector<std::string> extra;
	for (auto& kv : data) {
		bool listed = false;
		for (auto& row : keyRows) {
			size_t i = row.find_first_not_of('-');
			if (i != std::string::npos && row.compare(i, std::string::npos, kv.first) == 0) { listed = true; break; }
		}
		if (!listed) extra.push_back(kv.first);
	}
	std::sort(extra.begin(), extra.end());
	if (!extra.empty()) stream << std::endl;
	for (auto& key : extra) stream << SingleToString(key, true) << std::endl;

	stream << std::string(4, '\n');

	std::cout << stream.str();
}
