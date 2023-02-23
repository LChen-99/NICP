/*
 * @Author: LuoChen 1425523063@qq.com
 * @Date: 2023-02-16 18:29:37
 * @LastEditors: LuoChen 1425523063@qq.com
 * @LastEditTime: 2023-02-21 10:49:47
 * @FilePath: /nicp/nicp/nicp/hello.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <iostream>
#include "pclcloudconverter.h"
#include <pcl/io/pcd_io.h>


int main(int argc, char const *argv[]){
    std::cout << "123" << std::endl;
    nicp::CloudConverter converter;
    nicp::Cloud cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    if (pcl::io::loadPCDFile<pcl::PointXYZ> (argv[1], *pcl_cloud) == -1)
    {
        PCL_ERROR ("Couldn't read file  \n");
        return (-1);
    }
    for(int i = 0; i < pcl_cloud->points.size(); i++){
        pcl_cloud->points[i].z = 0;
    }
    converter.compute(cloud, pcl_cloud);

    return 0;
}