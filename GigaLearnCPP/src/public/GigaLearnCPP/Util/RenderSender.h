#pragma once
#include "Report.h"
#include <pybind11/pybind11.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/BasicTypes/Action.h>
#include <GigaLearnCPP/Util/Timer.h>

namespace GGL {
	struct RG_IMEXPORT RenderSender {
		pybind11::module pyMod;

		float timeScale;
		double adaptiveRenderDelay = -1;
		Timer renderTimer = {};

		RenderSender(float timeScale);

		RG_NO_COPY(RenderSender);

		// rewardNames/playerRewards (optional): per-player weighted reward breakdown, shown by the visualiser
		void Send(const RLGC::GameState& state,
			const std::vector<std::string>* rewardNames = nullptr, const std::vector<std::vector<float>>* playerRewards = nullptr);

		~RenderSender();
	};
}