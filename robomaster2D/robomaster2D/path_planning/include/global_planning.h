#include "build_map.h"
namespace hitcrt
{
	//自定义规划类
	class GlobalPlanner
	{
	public:
		DStarliteSearch<MapSearchNode> dstar;//多目标 D*Lite 搜索类
		vector<MapSearchNode> nodes;//节点列表
		vector<vector<MapSearchNode>> obstacles;//障碍物列表
		vector<vector<float>> c_path;//约束路径
		vector<float> last_endposition;//上次规划的目标位置
		uint last_endpoint;//上次规划的目标点
		vector<uint> path_index;//路径的节点索引列表
		vector<vector<float>> last_path;//上次规划的路径

		void PlannerInit();//规划器初始化
		void PlannerDelete();//删除节点信息
		void refreshObstacle();//重新初始化静态障碍物
		vector<vector<float>> PlannerStart(vector<float> StartPos, vector<float> EndPos);//开始规划，起始位置到目标位置
		void addObstacle(float x, float y, float width_x, float width_y);
		void PlannerAddPunish(uint Area);
		void PlannerAddObstacle(uint NumberA, uint NumberB);
	private:
		vector<MapSearchNode> static_nodes;
	};
}


